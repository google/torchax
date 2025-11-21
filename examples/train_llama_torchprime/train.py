# Utilities for training
import copy
import functools
import time

import jax
import optax
import torch
import torchax
from jax.experimental import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import tree_map
from torchax import interop

SEQLEN = 2048
GRADIENT_CLIP = 1.0

## Interface for optimizer_fn:
# def optimizer_fn(optimizer_state, weights, gradients) -> new_optimizer_state, new_weights

## Interface for loss_fn:
## loss_fn(model_out, label) -> scalar


class TraininableLlama:
  def __init__(self, model):
    self.orig_model = model

  # Args is what dataloader gives
  def call(self, weights, buffers, args, kwargs):
    weights_and_buffers = copy.copy(weights)
    weights_and_buffers.update(buffers)
    return torch.func.call_functional(
      self.orig_model, weights_and_buffers, args, kwargs
    )


def fake_dataloader(size, seqlen, batch_size):
  for _ in range(size):
    x = torch.randint(0, 32000, (batch_size, seqlen), device="cpu")
    yield x, (x + 1) % 32000


def group_data(dataloader, block_size):
  """yields tuple of inputs, label with seqlen == block_size"""

  tally = 0
  inputs = []
  labels = []

  for line in dataloader:
    x, y = line  # line['input_ids'], line['labels']
    inputs.append(x)
    labels.append(y)
    seqlen = x.shape[1]
    tally += seqlen
    if tally >= block_size:
      inputs_stacked = torch.concat(inputs, dim=-1)
      labels_stacked = torch.concat(labels, dim=-1)
      yield inputs_stacked, labels_stacked
      tally = 0
      inputs = []
      labels = []


def sharded_device_put(tensor, sharding):
  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)
  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  shape = tensor.shape
  x_split = [
    jax.device_put(tensor[i], device)
    for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


# NOTE: this line makes jax.remat able to take torch functions
remat = interop.torch_view(jax.remat)
mark_sharding = interop.torch_view(jax.lax.with_sharding_constraint)


def make_train_step(model_forward, loss_fn, optax_optimizer, policy):
  env = torchax.default_env()

  @functools.partial(remat, policy=policy)
  def loss(weights, args, label):  # inputs are XLATensor
    with env, jax.named_scope("compute_loss"):
      args = (mark_sharding(args[0], P("fsdp")), *args[1:])
      res = model_forward(weights, args)
      res = mark_sharding(res, P("fsdp"))
      num_tokens = res.shape[-1]
      flattened = res.reshape(-1, num_tokens)
      label = label.reshape(-1)
      loss = loss_fn(flattened, label)
      return loss

  jloss = interop.jax_view(loss)
  grad_fn = jax.value_and_grad(jloss)

  def step(weights, opt_state, args, label):  # inputs are array
    with jax.named_scope("compute_gradient"):
      loss, gradient = grad_fn(weights, args, label)

    with jax.named_scope("optimizer_updates"):
      updates, opt_state = optax_optimizer.update(gradient, opt_state, weights)
      weights = optax.apply_updates(weights, updates)
    return loss, weights, opt_state

  return step


def _prelower_step(step, weights, opt_state, args, label, mesh):
  wshardings = tree_map(
    lambda a: a.sharding if isinstance(a, jax.Array) else None, weights
  )
  oshardings = tree_map(
    lambda a: a.sharding if isinstance(a, jax.Array) else None, opt_state
  )

  print("Start compiling")
  start = time.perf_counter()
  lowered = jax.jit(
    step,
    donate_argnums=(0, 1),
    # in_shardings=shardings,
    out_shardings=(NamedSharding(mesh, P()), wshardings, oshardings),
  ).lower(weights, opt_state, args, label)
  # print(lowered.as_text())
  # import pdb; pdb.set_trace()
  print("program size:", len(lowered.as_text()) / 1e6, "m chars")
  step_compiled = lowered.compile()
  end = time.perf_counter()
  print("End compiling", end - start)
  compile_time = end - start
  print("Compile time:", compile_time)
  cost_analysis = step_compiled.cost_analysis()
  if isinstance(cost_analysis, dict):
    cost_analysis = [cost_analysis]
  for co in cost_analysis:
    print("Flops", co["flops"])
    print("GB accessed", co["bytes accessed"] / 1e9)
  return step_compiled


def train_loop(
  mesh,
  model,
  weights,
  data_loader,
  input_freqs_cis,
  lr,
  seqlen,
  policy,
  batch_size,
  use_shmap,
  profile_dir: str,
):
  print("start training")
  min_loop_time = 10000

  env = torchax.default_env()

  jax_params = env.t2j_iso(weights)
  jax_optimizer = optax.chain(
    optax.clip_by_global_norm(GRADIENT_CLIP),  # Clip gradients to a max norm of 1.0
    optax.sgd(lr),
  )
  opt_state = jax_optimizer.init(jax_params)
  # opt_state = (ScaleByAdamState(
  #   # replicate count
  #   jax.device_put(opt_state[0].count, NamedSharding(mesh, P())),
  #   opt_state[0].mu,
  #   opt_state[0].nu
  # ), *opt_state[1:])

  wspecs = tree_map(lambda a: a.sharding.spec, jax_params)

  model_forward_orig = functools.partial(torch.func.functional_call, model)

  @functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(wspecs, (P("fsdp"), P(), P(), P())),
    out_specs=(P("fsdp")),
    check_rep=False,
  )
  def model_forward_shmap(weight, args):
    def gather_weights(w, spec):
      try:
        index = spec.index("fsdp")
        w = jax.lax.all_gather(w, axis_name="fsdp", tiled=True, axis=index)
        return w
      except ValueError:
        return w

    new_weights = {}
    for k, v in weight.items():
      if not k.startswith("layers"):
        new_weights[k] = gather_weights(v, wspecs[k])
      else:
        new_weights[k] = v

    res = interop.call_torch(model_forward_orig, new_weights, args)
    return res

  if use_shmap:
    model_forward = interop.torch_view(model_forward_shmap)
  else:
    model_forward = model_forward_orig

  train_step = make_train_step(
    model_forward,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optax_optimizer=jax_optimizer,
    policy=policy,
  )

  def _expand_input(input_seq):
    seqlen = input_seq.shape[1]
    freqs_cis = env.t2j_iso(input_freqs_cis[:seqlen])
    mask = torch.full((seqlen, seqlen), float("-inf"), device="cpu")
    mask = torch.triu(mask, diagonal=1)
    return (input_seq, 0, freqs_cis, mask)

  replicated_sharding = NamedSharding(mesh, P())
  fsdp_sharding = NamedSharding(mesh, P("fsdp"))

  def _shard_first_dim(x):
    with jax.default_device(jax.devices("cpu")[0]):
      xj = env.to_xla(x).jax()

    xj = jax.make_array_from_callback(xj.shape, fsdp_sharding, lambda a: xj[a])
    return xj

  def _replicate(x):
    with jax.default_device(jax.devices("cpu")[0]):
      xj = env.to_xla(x).jax()
    xj = jax.make_array_from_callback(xj.shape, replicated_sharding, lambda a: xj)
    return xj

  data_iter = fake_dataloader(1000, seqlen, batch_size)

  for i, item in enumerate(data_iter):
    with jax.profiler.StepTraceAnnotation("train", step_num=i):
      inputs, labels = item

      input_seq, pos, freqs_cis, mask = _expand_input(inputs)

      input_seq = _shard_first_dim(input_seq)
      freqs_cis = freqs_cis
      mask = _replicate(mask)
      labels = _shard_first_dim(labels)

      # print("INPUT shape", inputs.shape)
      if i == 0:
        # NOTE: this is not necessary; but I want to print out
        # Stablehlo, and compile times
        train_step = _prelower_step(
          train_step,
          jax_params,
          opt_state,
          (input_seq, pos, freqs_cis, mask),
          labels,
          mesh,
        )

      if i == 5:
        jax.profiler.start_trace(profile_dir)
      step_start = time.perf_counter()
      loss, jax_params, opt_state = train_step(
        jax_params, opt_state, (input_seq, pos, freqs_cis, mask), labels
      )
      jax.block_until_ready((loss, jax_params))
      step_end = time.perf_counter()
      if i == 6:
        jax.profiler.stop_trace()

      print(
        "* Training step",
        i,
        "loss",
        loss,
        loss.dtype,
        "step latency: ",
        step_end - step_start,
      )
      min_loop_time = min(min_loop_time, step_end - step_start)
      print("====================")
      if i >= 6:
        break

  return min_loop_time
