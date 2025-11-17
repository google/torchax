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

"""Flax interop."""

import torch

import torchax as tx
import torchax.interop


class FlaxNNModule(torch.nn.Module):
  def __init__(self, env, flax_module, sample_args, sample_kwargs=None):
    super().__init__()
    prng = env.prng_key
    sample_kwargs = sample_kwargs or {}
    parameter_dict = tx.interop.call_jax(
      flax_module.init, prng, *sample_args, **sample_kwargs
    )

    self._params = self._encode_nested_dict(parameter_dict)

    self._flax_module = flax_module

  def _encode_nested_dict(self, nested_dict):
    child_module = torch.nn.Module()
    for k, v in nested_dict.items():
      if isinstance(v, dict):
        child_module.add_module(k, self._encode_nested_dict(v))
      else:
        child_module.register_parameter(k, torch.nn.Parameter(v))
    return child_module

  def _decode_nested_dict(self, child_module):
    result = dict(child_module.named_parameters(recurse=False))
    for k, v in child_module.named_children():
      result[k] = self._decode_nested_dict(v)
    return result

  def forward(self, *args, **kwargs):
    nested_dict_params = self._decode_nested_dict(self._params)
    return tx.interop.call_jax(
      self._flax_module.apply, nested_dict_params, *args, **kwargs
    )
