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
import torch
import torch.nn.functional as F
from torch.library import Library, impl, impl_abstract
import torchax
import torchax.export
from torchax.ops import jaten
from torchax.ops import jlibrary
# Create a `mylib` library which has a basic SDPA op.
m = Library("mylib", "DEF")
m.define("scaled_dot_product_attention(Tensor q, Tensor k, Tensor v) -> Tensor")


@impl(m, "scaled_dot_product_attention", "CompositeExplicitAutograd")
def _mylib_scaled_dot_product_attention(q, k, v):
  """Basic scaled dot product attention without all the flags/features."""
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  y = F.scaled_dot_product_attention(
      q,
      k,
      v,
      dropout_p=0,
      is_causal=False,
      scale=None,
  )
  return y.transpose(1, 2)


@impl_abstract("mylib::scaled_dot_product_attention")
def _mylib_scaled_dot_product_attention_meta(q, k, v):
  return torch.empty_like(q)


# Register library op as a composite for export using the `@impl` method
# for a torch decomposition.
jlibrary.register_torch_composite(
    "mylib.scaled_dot_product_attention", _mylib_scaled_dot_product_attention,
    torch.ops.mylib.scaled_dot_product_attention,
    torch.ops.mylib.scaled_dot_product_attention.default)

# Also register ATen softmax as a composite for export in the `mylib` library
# using the JAX ATen decomposition from `jaten`.
jlibrary.register_jax_composite(
    "mylib.softmax",
    jaten._aten_softmax,
    torch.ops.aten._softmax,
    static_argnums=1  # Required by JAX jit
)


class LibraryTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)

  def test_basic_sdpa_library(self):

    class CustomOpExample(torch.nn.Module):

      def forward(self, q, k, v):
        x = torch.ops.mylib.scaled_dot_product_attention(q, k, v)
        x = x + 1
        return x

    # Export and check for composite operations
    model = CustomOpExample()
    arg = torch.rand(32, 8, 128, 64)
    args = (
        arg,
        arg,
        arg,
    )

    exported = torch.export.export(model, args=args)
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())

    ## TODO Update this machinery from producing function calls to producing
    ## stablehlo.composite ops.
    self.assertIn("call @mylib.scaled_dot_product_attention", module_str)
    self.assertIn("call @mylib.softmax", module_str)


if __name__ == '__main__':
  unittest.main()
