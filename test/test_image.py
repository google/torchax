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

import itertools
import unittest
from functools import partial

import jax
import torch
from absl.testing import parameterized

import torchax
import torchax.interop


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def upsample_jit(
  tensor,
  output_size: tuple[int, int],
  align_corners: bool,
  antialias: bool,
  method: str,
):
  tensor = torchax.interop.torch_view(tensor)
  tensor = torch.nn.functional.interpolate(
    tensor,
    size=output_size,
    mode=method,
    align_corners=align_corners,
    antialias=antialias,
  )
  return torchax.interop.jax_view(tensor)


class TestResampling(parameterized.TestCase):
  @parameterized.product(
    antialias=[
      True,
      False,
    ],
    align_corners=[
      False,
      True,
    ],
  )
  def test_resampling_combinations_bicubic(self, antialias, align_corners):
    method = "bicubic"
    input_tensor = torch.rand((1, 1, 256, 512), dtype=torch.float32)
    output_size = (128, 64)

    upsampled_tensor = torch.nn.functional.interpolate(
      input_tensor,
      size=output_size,
      mode=method,
      align_corners=align_corners,
      antialias=antialias,
    )

    env = torchax.default_env()
    with env:
      input_tensor_xla = env.to_xla(input_tensor)
      input_tensor_xla = torchax.interop.jax_view(input_tensor_xla)
      upsampled_tensor_xla = upsample_jit(
        input_tensor_xla, output_size, align_corners, antialias=antialias, method=method
      )

    upsampled_tensor_xla = env.j2t_copy(upsampled_tensor_xla)
    abs_err = torch.abs(upsampled_tensor - upsampled_tensor_xla)

    assert torch.allclose(
      upsampled_tensor, upsampled_tensor_xla, atol=1e-4, rtol=1e-5
    ), f"{method} upsampling failed with error {abs_err.max()}"


if __name__ == "__main__":
  unittest.main()
