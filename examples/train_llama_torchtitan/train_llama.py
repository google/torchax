# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import logging
import zlib
from typing import Tuple
from collections import defaultdict
import functools
import numpy as np
import torch
import torch.nn.functional
from torch.utils import _pytree as pytree
import splash_attn
import helper

import torchax as tx
import torchax.interop
import torchax.train
from torchax.interop import jax_view, torch_view, JittableModule
import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
import optax

from torchtitan.models.llama3 import llama3_args
from torchtitan.models.llama3.model import model as titan

P = jax.sharding.PartitionSpec

# --- LOGGING HELPER ---
def log_msg(msg):
    """Logs on ALL processes to ensure errors on workers are seen."""
    # Including process index helps debug which host is doing what or failing
    print(f"[Proc {jax.process_index()}] {msg}", flush=True)


def sharded_device_put(tensor: jax.Array, sharding) -> jax.Array:

  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()

  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  # NOTE: at here, num_global_devices != num_local_devices
  # meaning we are in multi-host setup. Each host will run the same process
  # and each process only need to handle the devices accessible to this host.
  shape = tensor.shape
  x_split = [
    jax.device_put(tensor[i], device)
    for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


sharding_map_original = {
  "freqs_cis": (),  #  torch.complex64 (2048, 64)
  "tok_embeddings.weight": ("fsdp", "tp"),  #  torch.float32 (vocab_size, 4096)
  "layers.*.attention.wo.weight": ("fsdp", "tp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wq.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wk.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wv.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.feed_forward.w1.weight": ("tp", "fsdp"),  #  torch.float32 (11008, 4096)
  "layers.*.feed_forward.w2.weight": ("fsdp", "tp"),  #  torch.float32 (4096, 11008)
  "layers.*.feed_forward.w3.weight": ("tp", "fsdp"),  #  torch.float32 (11008, 4096)
  "layers.*.attention_norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "layers.*.ffn_norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "output.weight": ("tp", "fsdp"),  #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan = {
  "freqs_cis": (),
  "tok_embeddings.weight": ("tp", "fsdp"),
  "layers.params.attention___wo___weight": (None, "fsdp", "tp"),
  "layers.params.attention___wq___weight": (None, "tp", "fsdp"),
  "layers.params.attention___wk___weight": (None, "tp", "fsdp"),
  "layers.params.attention___wv___weight": (None, "tp", "fsdp"),
  "layers.params.feed_forward___w1___weight": (None, "tp", "fsdp"),
  "layers.params.feed_forward___w2___weight": (None, "fsdp", "tp"),
  "layers.params.feed_forward___w3___weight": (None, "tp", "fsdp"),
  "layers.params.attention_norm___weight": (None, "fsdp"),
  "layers.params.ffn_norm___weight": (None, "fsdp"),
  "norm.weight": ("fsdp",),
  "output.weight": ("tp", "fsdp"),
}


class Trainer:
  def __init__(self, mesh):
    self.mesh = mesh
    self.x_sharding = jax.sharding.NamedSharding(self.mesh, P("fsdp"))
    self.replicated = jax.sharding.NamedSharding(self.mesh, P())

  def fit(self, model, loss_fn, data_loader, train_steps=25):
    xla_env = torchax.default_env()
    jax.config.update("jax_enable_x64", False)
    xla_env._mesh = self.mesh
    xla_env.use_flash_attention = True

    jittable_mod = JittableModule(model)

    # split the params to the n devices

    # model_fn is responsible to shard if needed
    # to do FSDP one shards the first input args and output
    # on the batch dimension
    def model_fn(weights, buffers, args):
      return jittable_mod.functional_call("forward", weights, buffers, args)

    jax_optimizer = optax.sgd(0.01)
    opt_state = torch_view(jax_optimizer.init(jax_view(jittable_mod.params)))

    # opt_state = torchax.interop.call_jax(jax_optimizer.init, jittable_mod.params)

    train_step = torchax.train.make_train_step(
      model_fn,
      loss_fn,
      jax_optimizer,
      remat_policy=jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        "device", "pinned_host"
      ),
    )

    # Metrics tracking
    total_tokens_after_warmup = 0
    total_time_after_warmup = 0
    avg_throughput = 0.0
    warmup_steps = 2

    print("Begining training")
    s = time.perf_counter()
    jax.profiler.start_trace("/tmp/tensorboard")
    print("start training")
    min_loop_time = 10000
    for i, item in enumerate(data_loader):
      inputs, labels = item

      current_batch_size, current_seq_len = inputs.shape[0], inputs.shape[1]
      tokens_this_step = current_batch_size * current_seq_len

      # Sharding input data
      # Ensure inputs are on CPU before calling sharded_device_put to avoid OOM on device 0
      if inputs.device.type != 'cpu':
          inputs = inputs.cpu()
      if labels.device.type != 'cpu':
          labels = labels.cpu()

      jax_inputs = sharded_device_put(inputs.numpy(), self.x_sharding)
      jax_labels = sharded_device_put(labels.numpy(), self.x_sharding)

      inputs = torch_view(jax_inputs)
      labels = torch_view(jax_labels)

      if i == 0:
        train_step = helper.compile_step_func(
          train_step,
          jittable_mod.params,
          jittable_mod.buffers,
          opt_state,
          inputs,
          labels,
          self.mesh,
        )

      print("INPUT shape", inputs.shape)
      step_start = time.perf_counter()
      loss, jittable_mod.params, opt_state = train_step(
        jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels
      )
      # wait for iteration to finish to measure time
      torchax.interop.call_jax(jax.block_until_ready, (loss, jittable_mod.params))
      step_end = time.perf_counter()
      loop_time = step_end - step_start
      current_throughput = tokens_this_step / loop_time
      print(
        f"Step {i + 1}/{train_steps} | Loss: {loss.item():.4f} | Throughput:"
        f" {current_throughput:.2f} tokens/sec",
      )
      min_loop_time = min(min_loop_time, loop_time)
      print("======")

      if i >= warmup_steps:
        total_tokens_after_warmup += tokens_this_step
        total_time_after_warmup += loop_time
        avg_throughput = total_tokens_after_warmup / total_time_after_warmup

      if i >= train_steps - 1:
        break
    jax.profiler.stop_trace()

    print(f"\nThroughput (avg): {avg_throughput:.2f} tokens/s")

    return min_loop_time


def _process_sharding_name(name):
  """Replace integers in param name with *.

  Presumably all layers should have the same sharding.
  """

  def is_integer(t):
    try:
      int(t)
      return True
    # pylint: disable-next=all
    except:  # noqa: E722
      return False

  tokens = name.split(".")
  for i, t in enumerate(tokens):
    if is_integer(t):
      tokens[i] = "*"
  return ".".join(tokens)

def _make_weight_shard(weight_meta, slice_index):
    # 1. Determine shape
    shard_meta = weight_meta[slice_index]
    
    # 2. DETERMINISTIC SEEDING (Fix for silent failures)
    # Python's hash() is randomized by default in Python 3.
    # We use zlib.adler32 to ensure Host 0 and Host 1 generate same seed for same index.
    tuple_bytes = str(tuple((s.start, s.stop, s.step) for s in slice_index)).encode('utf-8')
    seed = zlib.adler32(tuple_bytes)
    
    log_msg(f"    - Init shard {slice_index} with seed {seed} (Shape: {shard_meta.shape})")
    key = jax.random.PRNGKey(seed)

    # 3. Dtype Mapping
    dtype_map = {
        torch.bfloat16: jnp.bfloat16,
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.complex64: jnp.complex64, 
        torch.complex128: jnp.complex128,
    }
    jax_dtype = dtype_map.get(shard_meta.dtype, jnp.bfloat16)

    # 4. Generate directly on device
    return jax.random.normal(key, shard_meta.shape, dtype=jax_dtype)


def create_sharded_weights(model, mesh, sharding_map):
  res = {}
  env = torchax.default_env()
  for name, weight_meta in model.state_dict().items():
    sharding_spec = sharding_map.get(_process_sharding_name(name))
    if sharding_spec is None:
      print("Skipping weight:", name)
      continue
    sharding = NamedSharding(mesh, P(*sharding_spec))
    print(f"Initializing weight {name} w shape={weight_meta.shape} dtype={weight_meta.dtype} sharding={sharding}....")
    res[name] = env.j2t_iso(
            jax.make_array_from_callback(
                weight_meta.shape,
                sharding,
                functools.partial(_make_weight_shard, weight_meta),
            )
        )
  return res


def fake_dataloader(size, seqlen, batch_size):
  for _ in range(size):
    x = torch.randint(0, 32000, (batch_size, seqlen), device="cpu")
    yield x, (x + 1) % 32000


def main(
  model_type="8B",
  batch_size=8,
  seqlen=2048,
  override_num_layers=-1,
  use_scan=True,
  tp_parallelism=1,
  train_steps=25,
):
  # 1. INIT DISTRIBUTED FIRST
  print(f"Process {os.getpid()} initializing JAX distributed...", flush=True)
  try:
      jax.distributed.initialize()
      print(f"Process {os.getpid()} JAX distributed initialized.", flush=True)
  except Exception as e:
      print(f"JAX distributed init skipped/failed (Normal for single host): {e}", flush=True)

  torchax.enable_globally()
  torchax.enable_performance_mode()


  num_global = jax.device_count()
  fsdp = num_global // tp_parallelism
  
  log_msg(f"Global Devices: {num_global}, Local Devices: {jax.local_device_count()}, Process: {jax.process_count()}")
  log_msg(f"Mesh Config: FSDP={fsdp}, TP={tp_parallelism}")

  # 2. MANUAL MESH CREATION (Fix for Mesh Utils Crash)
  # Using raw reshape is safer for multi-slice than mesh_utils
  devices = jax.devices()
  
  # Safety check to prevent silent reshaping errors
  if len(devices) != fsdp * tp_parallelism:
      raise ValueError(f"Total devices ({len(devices)}) must match FSDP*TP ({fsdp}*{tp_parallelism})")

  device_mesh_array = np.array(devices).reshape(fsdp, tp_parallelism)
  mesh = jax.sharding.Mesh(device_mesh_array, ("fsdp", "tp"))
  
  log_msg("Mesh successfully created.")

  if use_scan:
    # using scan the individial weights will have shape (num_layers, w, h)
    sharding_map = sharding_map_scan
  else:
    sharding_map = sharding_map_original

  log_msg(f"Jialei: creating env")
  env = torchax.default_env()
  env.config.use_tpu_flash_attention = True
  env.config.shmap_flash_attention = True
  env._mesh = mesh  # this is the mesh used by flash attention pallas kernel


  log_msg(f"Jialei: creating args")
  args = llama3_args[model_type]
  # Note: torchtitan's upstream config did not specify this value
  args.vocab_size = 128256
  args.max_seq_len = seqlen
  if override_num_layers > 0:
    args.n_layers = override_num_layers

  log_msg("Creating Meta Model (Patched for scalability)...")
  original_precompute = titan.Transformer._precompute_freqs_cis
  
  def dummy_precompute(self):
      # Returns a dummy tensor on meta device with correct shape/dtype
      # Avoids torch.polar/complex ops on meta backend
      head_dim = self.params.dim // self.params.n_heads
      return torch.empty(
          self.params.max_seq_len, 
          head_dim // 2, 
          dtype=torch.complex64, 
          device="meta"
      )

  # Apply patch
  titan.Transformer._precompute_freqs_cis = dummy_precompute
    
  # Note: because a single device don't have enough HBM memory
  # nor enough CPU memory to hold the parameters. We instantiate
  # the model on meta then manually initialize then shard each param
  torch.set_default_dtype(torch.bfloat16)
  with torch.device("meta"):
    gpt = titan.Transformer(args)

  log_msg(f'Model initialized with {sum(p.numel() for p in gpt.parameters())/1e9:.2f} B parameters.')

  # Restore original method immediately
  titan.Transformer._precompute_freqs_cis = original_precompute
  log_msg("Meta model created. Computing real freqs_cis on CPU...")

  with torch.device("cpu"):
    # need actual value for freqs_cis
    freqs_cis = gpt._precompute_freqs_cis()

  if use_scan:
    checkpoint_policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(
      "device", "pinned_host"
    )
    gpt = TransfomerWithScan(gpt, checkpoint_policy)

  log_msg(f"Jialei: creating weights")
  state_dict = dict(gpt.state_dict())
  state_dict.pop("freqs_cis")  # dont shard freqs_cis
  state_dict = create_sharded_weights(gpt, mesh, sharding_map)
  replicated = jax.sharding.NamedSharding(mesh, P())

  state_dict["freqs_cis"] = freqs_cis.to("jax").apply_jax(jax.device_put, replicated)
  gpt.load_state_dict(state_dict, assign=True)

  train_loader = fake_dataloader(train_steps, seqlen, batch_size)

  log_msg(f"Jialei: creating attention override")
  # NOTE: overriding attention to capture mesh and sharding info
  partition = P("fsdp", "tp", None, None)
  attention = functools.partial(splash_attn.tpu_splash_attention, mesh, partition, True)
  attention = jax.jit(attention)

  def custom_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
  ):
    #  batch, num of head, seq, dim
    jk, jq, jv = jax_view((query, key, value))
    res = attention(jk, jq, jv, None)
    return torch_view(res)

  log_msg(f"Jialei: actually creating attention override")
  env.override_op_definition(
    torch.nn.functional.scaled_dot_product_attention, custom_attention
  )

  def loss_fn(logits, y):
    num_tokens = logits.shape[-1]
    logits = logits.reshape(-1, num_tokens)
    y = y.reshape(-1)
    return torch.nn.functional.cross_entropy(logits, y)

  log_msg(f"Jialei: start training")
  with mesh:
    trainer = Trainer(mesh)
    return trainer.fit(gpt, loss_fn, train_loader)


class TransfomerWithScan(torch.nn.Module):
  def __init__(self, old_transformer, checkpoint_policy):
    super().__init__()
    self.tok_embeddings = old_transformer.tok_embeddings
    self.norm = old_transformer.norm
    self.output = old_transformer.output
    self.layers = torchax.train.ScannedModule(
      list(old_transformer.layers.values()), checkpoint_policy
    )

    self.register_buffer("freqs_cis", old_transformer.freqs_cis)

  def forward(self, tokens: torch.Tensor):
    """
    Perform a forward pass through the Transformer model.

    Args:
        tokens (torch.Tensor): Input token indices.

    Returns:
        torch.Tensor: Output logits after applying the Transformer model.

    """
    # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
    h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

    # for layer in self.layers.values():
    #     h = layer(h, self.freqs_cis)

    h = self.layers(h, self.freqs_cis, None)

    h = self.norm(h) if self.norm else h
    output = self.output(h) if self.output else h
    return output


if __name__ == "__main__":
  import fire
  try:
      fire.Fire(main)
  except Exception:
      # Catch crash and print traceback to stderr so it appears in logs
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)