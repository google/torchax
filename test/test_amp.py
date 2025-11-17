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

import unittest

import jax
import jax.numpy as jnp
import torch

import torchax
from torchax import interop


class AutocastTest(unittest.TestCase):
  def setUp(self):
    self.env = torchax.default_env()

  def test_auto_cast_ir(self):
    with self.env:
      with torchax.amp.autocast("jax", dtype=torch.bfloat16, env=self.env):
        a = jax.ShapeDtypeStruct((2, 2), jnp.float32)
        b = jax.ShapeDtypeStruct((2, 2), jnp.float32)
        ir_text = jax.jit(interop.jax_view(torch.matmul)).lower(a, b).as_text()
        self.assertIn("tensor<2x2xbf16>", ir_text)

  def test_auto_cast_matmul(self):
    with self.env:
      a = torch.randn(2, 2, device="jax")
      b = torch.randn(2, 2, device="jax")
      with torchax.amp.autocast("jax", dtype=torch.bfloat16, env=self.env):
        c = a @ b

      self.assertEqual(c.dtype, torch.bfloat16)

      with torch.autocast("cpu", dtype=torch.bfloat16):
        c_cpu = a.cpu() @ b.cpu()

      self.assertTrue(torch.allclose(c.cpu(), c_cpu))


if __name__ == "__main__":
  unittest.main()
