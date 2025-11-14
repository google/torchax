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
from typing import Tuple
from collections import defaultdict
import functools
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

num_global_devices = jax.device_count()
num_local_devices = jax.local_device_count()


def sharded_device_put(tensor: jax.Array, sharding) -> jax.Array:
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
    "tok_embeddings.weight":
        ('fsdp', 'tp'),  #  torch.float32 (vocab_size, 4096)
    "layers.*.attention.wo.weight": ('fsdp', 'tp'),  #  torch.int8 (4096, 4096)
    "layers.*.attention.wq.weight": ('tp', 'fsdp'),  #  torch.int8 (4096, 4096)
    "layers.*.attention.wk.weight": ('tp', 'fsdp'),  #  torch.int8 (4096, 4096)
    "layers.*.attention.wv.weight": ('tp', 'fsdp'),  #  torch.int8 (4096, 4096)
    "layers.*.feed_forward.w1.weight":
        ('tp', 'fsdp'),  #  torch.float32 (11008, 4096)
    "layers.*.feed_forward.w2.weight":
        ('fsdp', 'tp'),  #  torch.float32 (4096, 11008)
    "layers.*.feed_forward.w3.weight":
        ('tp', 'fsdp'),  #  torch.float32 (11008, 4096)
    "layers.*.attention_norm.weight": ('fsdp',),  #  torch.float32 (4096,)
    "layers.*.ffn_norm.weight": ('fsdp',),  #  torch.float32 (4096,)
    "norm.weight": ('fsdp',),  #  torch.float32 (4096,)
    "output.weight": ('tp', 'fsdp'),  #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan = {
    "freqs_cis": (),  #  torch.complex64 (2048, 64)
    # ParallelEmbedding for llama2; VocabParallelEmbedding for 3
    "tok_embeddings.weight":
        ('tp', 'fsdp'),  #  torch.float32 (vocab_size, 4096)
    "layers.params.attention___wo___weight":
        (None, 'fsdp', 'tp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.attention___wq___weight":
        (None, 'tp', 'fsdp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.attention___wk___weight":
        (None, 'tp', 'fsdp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.attention___wv___weight":
        (None, 'tp', 'fsdp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.feed_forward___w1___weight":
        (None, 'tp', 'fsdp'),  #  torch.float32 (n, 11008, 4096)
    "layers.params.feed_forward___w2___weight":
        (None, 'fsdp', 'tp'),  #  torch.float32 (n, 4096, 11008)
    "layers.params.feed_forward___w3___weight":
        (None, 'tp', 'fsdp'),  #  torch.float32 (n, 11008, 4096)
    "layers.params.attention_norm___weight": (
        None,
        'fsdp',
    ),  #  torch.float32 (n, 4096,)
    "layers.params.ffn_norm___weight": (
        None,
        'fsdp',
    ),  #  torch.float32 (n, 4096,)
    "norm.weight": ('fsdp',),  #  torch.float32 (4096,)
    "output.weight": ('tp', 'fsdp'),  #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan_fsdp = {
    "freqs_cis": (),  #  torch.complex64 (2048, 64)
    # ParallelEmbedding for llama2; VocabParallelEmbedding for 3
    "tok_embeddings.weight": ('fsdp',),  #  torch.float32 (vocab_size, 4096)
    "layers.params.attention___wo___weight":
        (None, 'fsdp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.attention___wq___weight":
        (None, 'fsdp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.attention___wk___weight":
        (None, 'fsdp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.attention___wv___weight":
        (None, 'fsdp'),  #  torch.int8 (n, 4096, 4096)
    "layers.params.feed_forward___w1___weight":
        (None, 'fsdp'),  #  torch.float32 (n, 11008, 4096)
    "layers.params.feed_forward___w2___weight":
        (None, 'fsdp'),  #  torch.float32 (n, 4096, 11008)
    "layers.params.feed_forward___w3___weight":
        (None, 'fsdp'),  #  torch.float32 (n, 11008, 4096)
    "layers.params.attention_norm___weight": (
        None,
        'fsdp',
    ),  #  torch.float32 (n, 4096,)
    "layers.params.ffn_norm___weight": (
        None,
        'fsdp',
    ),  #  torch.float32 (n, 4096,)
    "norm.weight": ('fsdp',),  #  torch.float32 (4096,)
    "output.weight": ('fsdp',),  #  torch.float32 (vocab_size, 4096)
}


class Trainer:

  def __init__(self, mesh):
    self.mesh = mesh
    self.x_sharding = jax.sharding.NamedSharding(self.mesh, P('fsdp'))
    self.replicated = jax.sharding.NamedSharding(self.mesh, P())

  def fit(self, model, loss_fn, data_loader, train_steps=25):
    xla_env = torchax.default_env()
    jax.config.update('jax_enable_x64', False)
    xla_env._mesh = self.mesh
    xla_env.use_flash_attention = True

    jittable_mod = JittableModule(model)

    # split the params to the n devices

    # model_fn is responsible to shard if needed
    # to do FSDP one shards the first input args and output
    # on the batch dimension
    def model_fn(weights, buffers, args):
      return jittable_mod.functional_call('forward', weights, buffers, args)

    jax_optimizer = optax.sgd(0.01)
    opt_state = torch_view(jax_optimizer.init(jax_view(jittable_mod.params)))

    #opt_state = torchax.interop.call_jax(jax_optimizer.init, jittable_mod.params)

    train_step = torchax.train.make_train_step(
        model_fn,
        loss_fn,
        jax_optimizer,
        remat_policy=jax.checkpoint_policies.offload_dot_with_no_batch_dims(
            'device', 'pinned_host'))


    # Metrics tracking
    total_tokens_after_warmup = 0
    total_time_after_warmup = 0
    avg_throughput = 0.0
    warmup_steps = 2

    print('Begining training')
    s = time.perf_counter()
    jax.profiler.start_trace('/tmp/tensorboard')
    print('start training')
    min_loop_time = 10000
    for i, item in enumerate(data_loader):
      inputs, labels = item

      current_batch_size, current_seq_len = inputs.shape[0], inputs.shape[1]
      tokens_this_step = current_batch_size * current_seq_len

      # Move them to jax device
      inputs = inputs.to('jax')
      labels = labels.to('jax')

      # Shard them on batch dim for fsdp
      inputs.apply_jax_(sharded_device_put, self.x_sharding)
      labels.apply_jax_(sharded_device_put, self.x_sharding)

      if i == 0:
        train_step = helper.compile_step_func(train_step, jittable_mod.params,
                                              jittable_mod.buffers, opt_state,
                                              inputs, labels, self.mesh)

      print('INPUT shape', inputs.shape)
      step_start = time.perf_counter()
      loss, jittable_mod.params, opt_state = train_step(jittable_mod.params,
                                                        jittable_mod.buffers,
                                                        opt_state, inputs,
                                                        labels)
      # wait for iteration to finish to measure time
      torchax.interop.call_jax(jax.block_until_ready,
                               (loss, jittable_mod.params))
      step_end = time.perf_counter()
      loop_time = step_end - step_start
      current_throughput = tokens_this_step / loop_time
      print(
          f'Step {i + 1}/{train_steps} | Loss: {loss.item():.4f} | Throughput:'
          f' {current_throughput:.2f} tokens/sec',
      )
      min_loop_time = min(min_loop_time, loop_time)
      print('======')

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
    # 1. Determine the specific shape for this shard
    shard_meta = weight_meta[slice_index]
    print(f".... working on shard at index={slice_index} with shape={shard_meta.shape}")

    # 2. Deterministic Seeding:
    # Generate a unique random key based on the slice_index.
    # This ensures every device initializes with different noise (or same if desired).
    # Note: Converting slice_index to a stable hash integer for seeding.
    seed = hash(tuple((s.start, s.stop, s.step) for s in slice_index)) % (2**31 - 1)
    key = jax.random.PRNGKey(seed)

    # 3. Handle Dtype Mapping (Torch -> JAX)
    # Ensure we allocate directly in bfloat16 if the model is bfloat16
    dtype_map = {
      torch.bfloat16: jnp.bfloat16,
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.complex64: jnp.complex64, 
        torch.complex128: jnp.complex128,
    }
    jax_dtype = dtype_map.get(shard_meta.dtype, jnp.bfloat16)

    # 4. Generate directly on the device using JAX
    # This allocates ONLY the memory needed for the shard, directly on XLA/TPU
    return jax.random.normal(key, shard_meta.shape, dtype=jax_dtype)


def create_sharded_weights(model, mesh, sharding_map):
  res = {}
  env = torchax.default_env()
  for name, weight_meta in model.state_dict().items():
    sharding_spec = sharding_map.get(_process_sharding_name(name))
    if sharding_spec is None:
      print('Skipping weight:', name)
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
    x = torch.randint(0, 32000, (batch_size, seqlen), device='cpu')
    yield x, (x + 1) % 32000


def main(
    model_type='8B',
    batch_size=8,
    seqlen=2048,
    override_num_layers=-1,
    use_scan=True,
    tp_parallelism=1,
    train_steps=25,
):
  torchax.enable_globally()
  torchax.enable_performance_mode()
  #logging.getLogger("jax").setLevel(logging.DEBUG)
  print(f"Running with parameters {locals()}")

  fsdp = num_global_devices // tp_parallelism
  mesh = jax.make_mesh((fsdp, tp_parallelism), ('fsdp', 'tp'))
  if use_scan:
    # using scan the individial weights will have shape (num_layers, w, h)
    sharding_map = sharding_map_scan
  else:
    sharding_map = sharding_map_original

  env = torchax.default_env()
  env.config.use_tpu_flash_attention = True
  env.config.shmap_flash_attention = True
  env._mesh = mesh  # this is the mesh used by flash attention pallas kernel

  args = llama3_args[model_type]
  # Note: torchtitan's upstream config did not specify this value
  args.vocab_size = 128256
  args.max_seq_len = seqlen
  if override_num_layers > 0:
    args.n_layers = override_num_layers

  # Note: because a single device don't have enough HBM memory
  # nor enough CPU memory to hold the parameters. We instantiate
  # the model on meta then manually initialize then shard each param
  torch.set_default_dtype(torch.bfloat16)
  with torch.device('meta'):
    gpt = titan.Transformer(args)
  print(f'Model initialized with {sum(p.numel() for p in gpt.parameters())/1e9:.2f} B parameters.')

  with torch.device('cpu'):
    # need actual value for freqs_cis
    freqs_cis = gpt._precompute_freqs_cis()

  if use_scan:
    checkpoint_policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        'device', 'pinned_host')
    gpt = TransfomerWithScan(gpt, checkpoint_policy)

  state_dict = dict(gpt.state_dict())
  state_dict.pop('freqs_cis')  # dont shard freqs_cis
  state_dict = create_sharded_weights(gpt, mesh, sharding_map)
  replicated = jax.sharding.NamedSharding(mesh, P())

  state_dict['freqs_cis'] = freqs_cis.to('jax').apply_jax(
      jax.device_put, replicated)
  gpt.load_state_dict(state_dict, assign=True)

  train_loader = fake_dataloader(train_steps, seqlen, batch_size)

  # NOTE: overriding attention to capture mesh and sharding info
  partition = P('fsdp', 'tp', None, None)
  attention = functools.partial(splash_attn.tpu_splash_attention, mesh,
                                partition, True)
  attention = jax.jit(attention)

  def custom_attention(query,
                       key,
                       value,
                       attn_mask=None,
                       dropout_p=0.0,
                       is_causal=False,
                       scale=None,
                       enable_gqa=False):
    #  batch, num of head, seq, dim
    jk, jq, jv = jax_view((query, key, value))
    res = attention(jk, jq, jv, None)
    return torch_view(res)

  env.override_op_definition(torch.nn.functional.scaled_dot_product_attention,
                             custom_attention)

  def loss_fn(logits, y):
    num_tokens = logits.shape[-1]
    logits = logits.reshape(-1, num_tokens)
    y = y.reshape(-1)
    return torch.nn.functional.cross_entropy(logits, y)

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
        list(old_transformer.layers.values()), checkpoint_policy)

    self.register_buffer('freqs_cis', old_transformer.freqs_cis)

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


if __name__ == '__main__':
  import fire
  fire.Fire(main)
