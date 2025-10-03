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

import torch
import torchax
import torchax.export
from . import base_test_util


class AddOne(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, a):
    return a + 1


class ConcatAddModel(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, a, b):
    a = torch.concat([a, a], dim=0)
    return a + b


class SymbolicShapeTest(base_test_util.TestCase):
  """Test possible symbolic shape computations that upstream torch export can
  emit. Seems to be currently limited to a few binary math operations where one
  operand is a symbolic variable/expr and the other is a constant integer.
  """

  def setUp(self):
    torch.manual_seed(0)

  def test_constraints_min_max(self):
    """Test a model with basic min/max dimension restrictions
    """

    # Arg shapes are a=s0{<=10}, b=s0*2
    model = AddOne()
    args = (torch.rand(5),)
    sym_a = torch.export.Dim("a", min=3, max=10)
    dynamic_shapes = ({0: sym_a},)

    with torch.no_grad():
      exported = torch.export.export(
          model, args=args, dynamic_shapes=dynamic_shapes)
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())

    self.assertRegex(module_str, r"stablehlo.constant.*3")
    self.assertRegex(module_str, r"shape_assertion.*s[0-9]+ >= 3")
    self.assertRegex(module_str, r"stablehlo.constant.*10")
    self.assertRegex(module_str, r"shape_assertion.*s[0-9]+ <= 10")

  def test_constraints_multiply(self):
    """Test a model with a slightly more complex constraint, where the input
    shapes are determined by an equation of the other, in this case input shapes
    are s0{<=10} and s0*2.
    """
    # Arg shapes are a=s0{<=10}, b=s0*2
    model = ConcatAddModel()
    args = (torch.rand(2), torch.rand(4))
    sym_a = torch.export.Dim("a", max=10)
    sym_b = sym_a * 2
    dynamic_shapes = ({0: sym_a}, {0: sym_b})

    with torch.no_grad():
      exported = torch.export.export(
          model, args=args, dynamic_shapes=dynamic_shapes)
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())

    self.assertRegex(module_str, r"stablehlo.constant.*10")
    self.assertRegex(module_str, r"shape_assertion.*s[0-9]+ <= 10")
    self.assertRegex(module_str, r"stablehlo.constant.*2")
    self.assertRegex(module_str, r"shape_assertion.*2\*s[0-9]+")

  def test_constraint_indirection(self):
    """Test a model where none of the shapes are directly symbolic variables
    but all are expressions of symints that don't appear directly in the model.
    """

    # Arg shapes are b=s0{<=10}*2
    args = (torch.randn(10, 10),)
    model = AddOne()
    sym_a = torch.export.Dim("a", max=10)
    sym_b = sym_a * 2
    dynamic_shapes = ({0: sym_b},)

    with torch.no_grad():
      exported = torch.export.export(
          model, args=args, dynamic_shapes=dynamic_shapes)
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())

    self.assertRegex(module_str, r"shape_assertion.*s[0-9]+ <= 10")
    self.assertRegex(module_str, r"shape_assertion.*2\*s[0-9]+")


if __name__ == "__main__":
  base_test_util.main()
