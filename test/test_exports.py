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
import jax.export
import torch
import torch.nn.functional as F
from packaging import version

import torchax
import torchax.export
from torchax import tensor
from torchax.ops import mappings


class Interpolate(torch.nn.Module):
  def forward(self, masks: torch.Tensor) -> torch.Tensor:
    masks = F.interpolate(
      masks,
      size=(500, 500),
      mode="bilinear",
      align_corners=False,
    )
    return masks


class TensorConstant(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, a):
    return a / torch.tensor(3)


class ExportTest(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)
    torchax.enable_accuracy_mode()

  def test_interpolate(self):
    # Check Accuracy
    arg = (torch.randn(3, 3, 200, 200),)
    model = Interpolate()
    ans = model(*arg)

    env = torchax.default_env()

    with torch.no_grad():
      exported = torch.export.export(model, arg)
    weights, func = torchax.export.exported_program_to_jax(exported)
    argj = env.t2j_copy(arg[0])
    ans2 = jax.jit(func)(weights, (argj,))[0]
    ans2 = env.j2t_copy(ans2)
    self.assertTrue(torch.allclose(ans, ans2, atol=1e-3))

    # Convert to StableHLO
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())
    self.assertIn("func.func public @main", module_str)
    self.assertIn("func.func private @clip(%arg0: tensor<500xf32>", module_str)
    self.assertIn("stablehlo.minimum", module_str)

  def test_constant(self):
    # Check Accuracy
    arg = (torch.randn(10, 10),)
    model = TensorConstant()
    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)
    env = torchax.default_env()
    weights, func = torchax.export.exported_program_to_jax(exported)
    argj = env.t2j_copy(arg[0])
    ans2 = jax.jit(func)(weights, (argj,))[0]
    ans2 = env.j2t_copy(ans2)
    self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))

    # Convert to StableHLO
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())
    self.assertIn("func.func public @main", module_str)
    self.assertIn("stablehlo.divide", module_str)

  def test_interpolate_dynamic(self):
    # Export with dynamic dimension constraints on both min and max
    arg = (torch.randn(3, 3, 200, 200),)
    model = Interpolate()
    model(*arg)
    dynamic_shapes = ({0: torch.export.Dim("b", min=3, max=10)},)

    with torch.no_grad():
      exported = torch.export.export(model, arg, dynamic_shapes=dynamic_shapes)
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())

    # Look for dynamic shape artifacts
    self.assertIn("func.func public @main(%arg0: tensor<?x3x200x200xf32>", module_str)
    self.assertIn("stablehlo.dynamic_broadcast_in_dim", module_str)
    self.assertIn("stablehlo.dynamic_gather", module_str)

  def test_export_dtypes(self):
    DTYPE_TO_MLIR_STR = {
      # NO_MAPPING        : jnp.float0 (signless scalar int)
      torch.bool: "i1",
      # NO_MAPPING        : "i4"
      torch.int8: "i8",
      torch.int16: "i16",
      torch.int32: "i32",
      torch.int64: "i64",
      torch.long: "i64",
      # NO_MAPPING        : "ui4"
      torch.uint8: "ui8",
      # NOTE(qihqi): torch export for uint16 seems broken at torch 2.4
      # torch.uint16        : "ui16",
      torch.uint32: "ui32",
      torch.uint64: "ui64",
      # NO_MAPPING        : "f8E4M3B11FNUZ"
      torch.float8_e4m3fn: "f8E4M3FN",
      # NO_MAPPING        : f8E4M3FNUZ
      torch.float8_e5m2: "f8E5M2",
      # NO_MAPPING        : f8E5M2FNUZ
      torch.bfloat16: "bf16",
      torch.half: "f16",
      torch.float16: "f16",
      torch.float32: "f32",
      torch.float64: "f64",
      torch.double: "f64",
      torch.complex64: "complex<f32>",
      torch.complex128: "complex<f64>",
      None: None,
    }

    model = TensorConstant()
    for torch_dtype in DTYPE_TO_MLIR_STR.keys():
      if torch_dtype is None:
        ## TODO: Figure out what the None mapping should be, seems like:
        ##   torch.tensor(dtype=None) maps to f32
        ##   jnp.tensor(dtype=None) maps to f64
        continue
      arg = (torch.randn(10).to(torch_dtype),)
      with torch.no_grad():
        exported = torch.export.export(model, arg)
      weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
      module_str = str(stablehlo.mlir_module())
      self.assertIn(DTYPE_TO_MLIR_STR[torch_dtype], module_str)

  def test_export_vhlo_target_version(self):
    arg = (torch.randn(10, 10),)
    model = TensorConstant()
    with torch.no_grad():
      exported = torch.export.export(model, arg)

    # 1. Output format "stablehlo" with target_version
    weights, exp_obj = torchax.export.exported_program_to_stablehlo(
      exported, target_version="1.0.0", output_format="stablehlo"
    )
    module_str = str(exp_obj.mlir_module())
    self.assertIn("func.func public @main", module_str)
    self.assertIn("stablehlo.divide", module_str)
    self.assertIn(b"ML\xefR", exp_obj.mlir_module_serialized)
    self.assertIn(b"StableHLO_v1.0.0", exp_obj.mlir_module_serialized)

    # 2. Output format "bytecode" with target_version
    weights, bytecode = torchax.export.exported_program_to_stablehlo(
      exported, target_version="1.0.0", output_format="bytecode"
    )
    self.assertIsInstance(bytecode, bytes)
    self.assertIn(b"ML\xefR", bytecode)
    self.assertIn(b"StableHLO_v1.0.0", bytecode)

    # 3. Output format "text" with target_version
    weights, vhlo_text = torchax.export.exported_program_to_stablehlo(
      exported, target_version="1.0.0", output_format="text"
    )
    self.assertIsInstance(vhlo_text, str)
    self.assertIn("vhlo.func_v1", vhlo_text)
    self.assertIn("vhlo.divide_v1", vhlo_text)

  def test_export_vhlo_versions(self):
    arg = (torch.randn(10, 10),)
    model = TensorConstant()
    with torch.no_grad():
      exported = torch.export.export(model, arg)

    for version_num in ("0.19.0", "1.0.0"):
      weights, bytecode = torchax.export.exported_program_to_stablehlo(
        exported, target_version=version_num, output_format="bytecode"
      )
      self.assertIsInstance(bytecode, bytes)
      self.assertIn(b"ML\xefR", bytecode)
      self.assertIn(f"StableHLO_v{version_num}".encode(), bytecode)

  @unittest.skipIf(
    version.parse(jax.__version__) < version.parse("0.11.1"),
    "Skipping VHLO deserialization test on older JAX versions",
  )
  def test_deserialize_vhlo_artifact(self):
    arg = (torch.randn(5, 5),)
    model = TensorConstant()
    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)

    # Test bytecode deserialization and execution
    weights, bytecode = torchax.export.exported_program_to_stablehlo(
      exported, target_version="1.0.0", output_format="bytecode"
    )
    deserialized_bc = torchax.export.deserialize_vhlo_artifact(bytecode)
    self.assertIn("func.func public @main", str(deserialized_bc.mlir_module()))
    self.assertIn("stablehlo.divide", str(deserialized_bc.mlir_module()))
    env = torchax.default_env()
    res_bc = deserialized_bc.call(weights, (env.t2j_copy(arg[0]),))
    res_bc_torch = env.j2t_copy(res_bc)
    self.assertTrue(torch.allclose(ans, res_bc_torch, atol=1e-5))

    # Test text deserialization and execution
    weights, vhlo_text = torchax.export.exported_program_to_stablehlo(
      exported, target_version="1.0.0", output_format="text"
    )
    deserialized_txt = torchax.export.deserialize_vhlo_artifact(vhlo_text)
    self.assertIn("func.func public @main", str(deserialized_txt.mlir_module()))
    self.assertIn("stablehlo.divide", str(deserialized_txt.mlir_module()))
    res_txt = deserialized_txt.call(weights, (env.t2j_copy(arg[0]),))
    res_txt_torch = env.j2t_copy(res_txt)
    self.assertTrue(torch.allclose(ans, res_txt_torch, atol=1e-5))

  def test_invalid_target_version_error(self):
    arg = (torch.randn(2, 2),)
    model = TensorConstant()
    with torch.no_grad():
      exported = torch.export.export(model, arg)

    # Invalid version format
    with self.assertRaises(ValueError) as ctx:
      torchax.export.exported_program_to_stablehlo(
        exported, target_version="invalid_ver"
      )
    self.assertIn("Invalid target_version format", str(ctx.exception))

    with self.assertRaises(ValueError) as ctx:
      torchax.export.exported_program_to_stablehlo(exported, target_version="1.0")
    self.assertIn("Invalid target_version format", str(ctx.exception))

    # Unsupported old version
    with self.assertRaises(ValueError) as ctx:
      torchax.export.exported_program_to_stablehlo(exported, target_version="0.0.1")
    self.assertIn("Unsupported target_version", str(ctx.exception))

    # Unsupported future version
    with self.assertRaises(ValueError) as ctx:
      torchax.export.exported_program_to_stablehlo(exported, target_version="999.0.0")
    self.assertIn("Unsupported target_version", str(ctx.exception))

    # Invalid type
    with self.assertRaises(TypeError):
      torchax.export.exported_program_to_stablehlo(exported, target_version=123)

    # Invalid output format
    with self.assertRaises(ValueError) as ctx:
      torchax.export.exported_program_to_stablehlo(
        exported, output_format="invalid_fmt"
      )
    self.assertIn("Unsupported output_format", str(ctx.exception))


if __name__ == "__main__":
  unittest.main()
