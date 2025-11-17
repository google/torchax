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

import jax
import jax.numpy as jnp
import torch

from torchax.ops import ops_registry


def op(*aten, **kwargs):
  def inner(func):
    for a in aten:
      ops_registry.register_torch_dispatch_op(a, func, **kwargs)
    return func

  return inner


@op(torch.ops._c10d_functional.all_gather_into_tensor)
def _c10d_all_gather(input, group_size: int, group_name: str):
  return jax.lax.all_gather(input, "torch_dist")


@op(torch.ops._c10d_functional.all_reduce)
def _c10d_all_reduce(self, reduceOp: str, group_name: str):
  if reduceOp == "sum":
    res = jax.lax.psum(self, axis_name="torch_dist")
  elif reduceOp == "avg":
    res = jax.lax.pmean(self, axis_name="torch_dist")
  elif reduceOp == "min":
    res = jax.lax.pmin(self, axis_name="torch_dist")
  elif reduceOp == "max":
    res = jax.lax.pmax(self, axis_name="torch_dist")
  else:
    raise RuntimeError(f"Reduce op {reduceOp} not implemented")
  return res


@op(torch.ops._c10d_functional.broadcast)
def _c10d_broadcast(self, src: int, group_name: str):
  masked = jnp.where(
    jax.lax.axis_index("torch_dist") == src,
    self,
    jnp.zeros_like(self),
  )
  return jax.lax.psum(masked, "torch_dist")


@op(torch.ops._c10d_functional.wait_tensor)
def _c10d_wait_tensor(tensor):
  # Async tensor is aleady `wait`ed by dispatcher
  return tensor
