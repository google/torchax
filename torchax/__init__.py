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

import contextlib
import dataclasses
import os
from contextlib import contextmanager
from typing import Any

import jax
import torch
from torch.utils import _pytree as pytree

import torchax.device_module
from torchax import tensor
from torchax.interop import JittableModule

from .checkpoint import load_checkpoint, save_checkpoint

__version__ = "0.0.12"
VERSION = __version__

# the "fast path" uses some sparse tensor thingies that currently we
# don't support
torch.backends.mha.set_fastpath_enabled(False)


__all__ = [
  "default_env",
  "extract_jax",
  "enable_globally",
  "save_checkpoint",
  "load_checkpoint",
]


os.environ.setdefault("ENABLE_RUNTIME_UPTIME_TELEMETRY", "1")

# torchax:oss-begin
if getattr(jax.config, "jax_pjrt_client_create_options", None):
  jax.config.update(
    "jax_pjrt_client_create_options",
    f"ml_framework_name:PyTorch/XLA2;ml_framework_version:{'v0.0.1'}",
  )
# torchax:oss-end

env = None


def default_env():
  global env

  if env is None:
    env = tensor.Environment()
  return env


def extract_jax(mod: torch.nn.Module, env=None, *, dedup_parameters=True):
  """Returns a pytree of jax.ndarray and a jax callable."""
  if env is None:
    env = default_env()

  jit_module = JittableModule(mod, dedup_parameters=dedup_parameters)

  states = dict(jit_module.buffers)
  states.update(jit_module.params)
  states = env.t2j_copy(states)

  def jax_func(states, args, kwargs=None):
    if kwargs is None:
      kwargs = {}

    # Pick the params and buffers supplied to jax_func
    params = {k: states[k] for k in jit_module.params.keys()}
    buffers = {k: states[k] for k in jit_module.buffers.keys()}
    (params, buffers, args, kwargs) = env.j2t_iso((params, buffers, args, kwargs))

    with env:
      res = jit_module.functional_call("forward", params, buffers, *args, **kwargs)

    return env.t2j_iso(res)

  return states, jax_func


def enable_globally():
  env = default_env().enable_torch_modes()
  return env


def disable_globally():
  global env
  default_env().disable_torch_modes()


@contextlib.contextmanager
def disable_temporarily():
  prev = default_env().enabled
  if prev:
    disable_globally()
  yield ()
  if prev:
    enable_globally()


torch.utils.rename_privateuse1_backend("jax")
unsupported_dtype = [torch.quint8]


torch._register_device_module("jax", torchax.device_module)


def enable_accuracy_mode():
  jax.config.update("jax_enable_x64", True)
  jax.config.update("jax_default_matmul_precision", "highest")
  default_env().config.internal_respect_torch_return_dtypes = True


def enable_performance_mode():
  jax.config.update("jax_enable_x64", False)
  jax.config.update("jax_default_matmul_precision", "default")
  default_env().config.internal_respect_torch_return_dtypes = False


@dataclasses.dataclass
class CompileOptions:
  # only valid if compiling nn.Module
  methods_to_compile: list[str] = dataclasses.field(default_factory=lambda: ["forward"])
  jax_jit_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  mode: str = "jax"  # or dynamo or export


def compile(fn, options: CompileOptions | None = None):
  options = options or CompileOptions()
  if options.mode == "jax":
    from torchax import interop

    if isinstance(fn, torch.nn.Module):
      module = interop.JittableModule(fn, extra_jit_args=options.jax_jit_kwargs)
      for n in options.methods_to_compile:
        module.make_jitted(n)
      return module
    else:
      return interop.jax_jit(fn)
  elif options.mode == "dynamo":
    raise RuntimeError("dynamo mode is not supported yet")
  elif options.mode == "export":
    raise RuntimeError("export mode is not supported yet")
