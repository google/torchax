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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.training import checkpoints

from . import tensor


def _to_jax(pytree):
  def to_jax_array(x):
    if isinstance(x, tensor.Tensor):
      return x.jax()
    elif isinstance(x, torch.Tensor):
      return jnp.asarray(x.cpu().numpy())
    return x

  return jax.tree_util.tree_map(to_jax_array, pytree)


def _to_torch(pytree):
  return jax.tree_util.tree_map(
    lambda x: torch.from_numpy(np.asarray(x))
    if isinstance(x, (jnp.ndarray, jax.Array))
    else x,
    pytree,
  )


def save_checkpoint(state: dict[str, Any], path: str, step: int):
  """Saves a checkpoint to a file in JAX style.

  Args:
    state: A dictionary containing the state to save. torch.Tensors will be
      converted to jax.Array.
    path: The path to save the checkpoint to. This is a directory.
    step: The training step.
  """
  state = _to_jax(state)
  checkpoints.save_checkpoint(path, state, step=step, overwrite=True)


def load_checkpoint(path: str) -> dict[str, Any]:
  """Loads a checkpoint and returns it in JAX format.

  This function can load both PyTorch-style (single file) and JAX-style
  (directory) checkpoints.

  If the checkpoint is in PyTorch format, it will be converted to JAX format.

  Args:
    path: The path to the checkpoint.

  Returns:
    The loaded state in JAX format (pytree with jax.Array leaves).
  """
  if os.path.isdir(path):
    # JAX-style checkpoint
    state = checkpoints.restore_checkpoint(path, target=None)
    if state is None:
      raise FileNotFoundError(f"No checkpoint found at {path}")
    return state
  elif os.path.isfile(path):
    # PyTorch-style checkpoint
    state = torch.load(path, weights_only=False)
    return _to_jax(state)
  else:
    raise FileNotFoundError(f"No such file or directory: {path}")
