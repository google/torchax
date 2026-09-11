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

# pylint: disable
"""Utilities for exporting a torch program to jax/stablehlo."""

import copy
import dataclasses
import io
import re
from typing import Any

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import sympy
import torch
import torch._refs
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager as pm
from jax._src.lib.mlir.dialects import hlo as stablehlo_dialect
from torch._decomp import get_decompositions
from torch.utils import _pytree as pytree

import torchax
from torchax import decompositions
from torchax.ops import mappings, ops_registry

DEBUG = False


class JaxInterpreter(torch.fx.Interpreter):
  """Experimental."""

  def __init__(self, graph_module):
    super().__init__(graph_module)

  def call_function(self, target, args: tuple, kwargs: dict) -> Any:
    if not isinstance(target, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
      return super().call_function(target, args, kwargs)

    if DEBUG:
      print("Running ", target.name(), "--------")

    op = ops_registry.all_aten_ops.get(target)
    if op is None:
      op = ops_registry.all_aten_ops.get(target.overloadpacket)
    assert op is not None, target
    assert op.is_jax_function, op
    if op is None:
      op = ops_registry.all_aten_ops.get(target.overloadpacket)
    if op is None:
      print(target.name(), target.tags)
      raise RuntimeError("No lowering found for", target.name())
    return op.func(*args, **kwargs)

  def run_node(self, n) -> Any:
    res = super().run_node(n)
    if DEBUG:
      if n.op == "call_function":
        if hasattr(res, "shape"):
          print("Meta:", n.meta.get("val").shape, "REAL: ", res.shape)
    return res


_extra_decomp = get_decompositions([torch.ops.aten.unfold])


def _extract_states_from_exported_program(exported_model):
  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = (
    exported_model.graph_signature.parameters + exported_model.graph_signature.buffers
  )
  state_dict = copy.copy(exported_model.state_dict)
  if (constants := getattr(exported_model, "constants", None)) is not None:
    state_dict.update(constants)
  param_buffer_values = [state_dict[key] for key in param_and_buffer_keys]

  if hasattr(exported_model.graph_signature, "lifted_tensor_constants"):
    for name in exported_model.graph_signature.lifted_tensor_constants:
      param_buffer_values.append(exported_model.tensor_constants[name])

  return param_and_buffer_keys, param_buffer_values


def exported_program_to_jax(exported_program, export_raw: bool = False):
  """returns a pytree of jax arrays(state), and

  a callable(func) that is jax function.

  func(state, input) would be how you call it.
  """
  if torch.__version__ >= "2.2":
    # torch version 2.1 didn't expose this yet
    exported_program = exported_program.run_decompositions()
    exported_program = exported_program.run_decompositions(
      decompositions.DECOMPOSITIONS
    )
  if DEBUG:
    print(exported_program.graph_module.code)

  names, states = _extract_states_from_exported_program(exported_program)

  def _extract_args(args, kwargs):
    flat_args, received_spec = pytree.tree_flatten((args, kwargs))  # type: ignore[possibly-undefined]
    return flat_args

  num_mutations = len(exported_program.graph_signature.buffers_to_mutate)

  def func(states, inputs):
    args = _extract_args(inputs, {})
    res = JaxInterpreter(exported_program.graph_module).run(
      *states,
      *args,
      enable_io_processing=False,
    )
    res = res[num_mutations:]
    return res

  if export_raw:
    return names, states, func
  env = torchax.default_env()
  states = env.t2j_copy(states)
  return states, func


def extract_avals(exported):
  """Return JAX Abstract Value shapes for all input parameters of the exported
  program. This supports dynamic batch dimensions, including with constraints.
  """

  def _to_aval(arg_meta, symbolic_shapes):
    """Convet from torch type to jax abstract value for export tracing"""

    def _get_dim(d):
      if isinstance(d, torch.SymInt):
        return symbolic_shapes[str(d)]
      return d

    val = arg_meta["val"]
    is_scalar = isinstance(val, float) or isinstance(val, int) or isinstance(val, bool)
    if is_scalar:
      return jax.ShapeDtypeStruct([], type(arg_meta["val"]))

    tensor_meta = arg_meta["tensor_meta"]
    shape = [_get_dim(d) for d in tensor_meta.shape]
    return jax.ShapeDtypeStruct(shape, mappings.t2j_dtype(tensor_meta.dtype))

  def _get_inputs(exported):
    """Return placeholders with input metadata"""
    placeholders = [p for p in exported.graph.nodes if p.op == "placeholder"]
    input_placeholders = [
      p
      for p, s in zip(placeholders, exported.graph_signature.input_specs, strict=False)
      if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
    ]
    return input_placeholders

  def _build_symbolic_shapes(range_constraints):
    """Convert torch SymInt to JAX symbolic_shape and stores in a map using the
    string name of the torch symbolic int.

    TODO: There is probably a better way of storing a key for a symbolic int.
    This value needs to be looked up again in `_to_aval` to figure out which
    JAX symbolic to map to for a given torch tensor.
    """
    if len(range_constraints) == 0:
      return None

    def _build_symbolic_constraints(symbol_name, torch_constraint):
      """Convert torch SymInt constraints to string for JAX symbolic_shape
      Using sympy may be overkill here, currently PyTorch only uses ValueRanges
      which allow specifying the min and the max of a value, for example:
        torch.export.Dim("a", min=5, max=10)
         ==> ("a >= 5", "a <= 10",)
      """
      if (
        not isinstance(torch_constraint, torch.utils._sympy.value_ranges.ValueRanges)
        or torch_constraint.is_bool
      ):
        raise TypeError(f"No symbolic constraint handler for: {torch_constraint}")

      constraints = []
      symbol = sympy.Symbol(symbol_name)
      if torch_constraint.lower != 2:
        constraints.append(symbol >= torch_constraint.lower)
      from sympy.core.singleton import S

      if (
        not torch_constraint.upper.is_infinite
        and torch_constraint.upper is not S.IntInfinity
      ):
        constraints.append(symbol <= torch_constraint.upper)

      return tuple(sympy.pretty(c, use_unicode=False) for c in constraints)

    def _build_symbolic_shape(sym, constraint, free_symbols):
      """Returns a JAX symbolic shape for a given symbol and constraint

      There are two possible sympy `sym` inputs:
        1. Symbol - (s0) These can have custom constraints.
        2. Expr - (s0*2) These apply the expr to s0's constraints, cannot override.

        Currently support is limited to operations with a symbol and and int,
        in `torch/export/dynamic_shapes.py`:
        "Only increasing linear operations with integer coefficients are supported."
      """
      symbol_name = str(sym)
      constraints = _build_symbolic_constraints(symbol_name, constraint)
      if sym.is_symbol:
        symbolic_shape = jax.export.symbolic_shape(symbol_name, constraints=constraints)
      else:
        assert len(sym.free_symbols) > 0
        scope = free_symbols[str(list(sym.free_symbols)[0])].scope
        symbolic_shape = jax.export.symbolic_shape(symbol_name, scope=scope)
      assert len(symbolic_shape) == 1
      return symbolic_shape[0]

    # Populate symbol variables before expressions, exprs need to use the same
    # Symbolic scope as the variable they operate on. Expressions can only be
    # integer compuations on symbol variables, so each symbol variable is OK to
    # have its own scope.
    symbolic_shapes = {}
    symbol_variables = [(s, v) for s, v in range_constraints.items() if s.is_symbol]
    symbol_exprs = [(s, v) for s, v in range_constraints.items() if not s.is_symbol]
    for sym, constraint in symbol_variables + symbol_exprs:
      symbolic_shape = _build_symbolic_shape(sym, constraint, symbolic_shapes)
      symbolic_shapes[str(sym)] = symbolic_shape
    return symbolic_shapes

  symbolic_shapes = _build_symbolic_shapes(exported.range_constraints)
  args = _get_inputs(exported)

  if DEBUG:
    print("Inputs to aval:", args, "--------")
    print("Symbolic shapes:", symbolic_shapes)
    for arg in args:
      print("Meta2Aval", arg.meta, "--> ", _to_aval(arg.meta, symbolic_shapes))

  return [_to_aval(arg.meta, symbolic_shapes) for arg in args]


def validate_target_version(target_version: str) -> None:
  """Validates that target_version is a valid StableHLO/VHLO version string and is supported."""
  if not isinstance(target_version, str):
    raise TypeError(
      f"target_version must be a string, got {type(target_version).__name__}"
    )

  target_version = target_version.strip()
  if not re.match(r"^\d+\.\d+\.\d+$", target_version):
    raise ValueError(
      f"Invalid target_version format: '{target_version}'. Expected format 'X.Y.Z' (e.g., '1.0.0')."
    )

  try:
    min_version = stablehlo_dialect.get_minimum_version()
    curr_version = stablehlo_dialect.get_current_version()
  except Exception:
    return

  try:
    if (
      stablehlo_dialect.get_smaller_version(target_version, min_version) != min_version
    ):
      raise ValueError(
        f"Unsupported target_version '{target_version}'. Version is older than "
        f"minimum supported version '{min_version}'."
      )
    if (
      stablehlo_dialect.get_smaller_version(target_version, curr_version)
      != target_version
    ):
      raise ValueError(
        f"Unsupported target_version '{target_version}'. Version is newer than "
        f"current version '{curr_version}'."
      )
  except ValueError:
    raise
  except Exception as e:
    raise ValueError(
      f"Invalid or unsupported target_version '{target_version}': {e}"
    ) from e


def legalize_stablehlo_to_vhlo(
  module_or_str: str | ir.Module,
  target_version: str | None = None,
  output_format: str = "bytecode",
) -> bytes | str:
  """Legalizes a StableHLO module to VHLO at the specified target version."""
  if target_version is not None:
    validate_target_version(target_version)
    target = target_version.strip()
  else:
    target = stablehlo_dialect.get_current_version()

  fmt = output_format.lower()
  if fmt in ("bytecode", "vhlo", "portable_artifact", "vhlo_bytecode"):
    try:
      if isinstance(module_or_str, str):
        return _jax.mlir.serialize_portable_artifact(module_or_str, target)
      else:
        return _jax.mlir.serialize_portable_artifact(str(module_or_str), target)
    except Exception as e:
      raise RuntimeError(
        f"Failed to serialize StableHLO module to VHLO bytecode for target_version '{target}': {e}"
      ) from e
  elif fmt in ("text", "mlir_text", "vhlo_text", "mlir"):
    try:
      stablehlo_dialect.register_stablehlo_passes()
      with jax_mlir.make_ir_context() as context:
        if isinstance(module_or_str, str):
          module = ir.Module.parse(module_or_str, context=context)
        else:
          module = ir.Module.parse(str(module_or_str), context=context)

        pipeline = f"builtin.module(stablehlo-legalize-to-vhlo{{allow-other-dialects=true}},vhlo-to-version{{target={target}}})"
        pass_manager = pm.PassManager.parse(pipeline, context=context)
        pass_manager.run(module.operation)
        return str(module)
    except Exception as e:
      raise RuntimeError(
        f"Failed to legalize StableHLO module to VHLO MLIR text for target_version '{target}': {e}"
      ) from e
  else:
    raise ValueError(
      f"Unsupported output_format: '{output_format}'. Supported formats: "
      "'stablehlo', 'bytecode', 'text', 'vhlo', 'mlir_text'."
    )


class DeserializedStableHLO:
  """Executable wrapper around a deserialized StableHLO MLIR module."""

  def __init__(
    self,
    module: ir.Module,
    bytecode: bytes | None = None,
    text: str | None = None,
  ):
    self._module = module
    self._context = module.context
    self._bytecode = bytecode
    self._text = text
    self._compiled = None

  @property
  def module(self) -> ir.Module:
    return self._module

  @property
  def mlir_module_serialized(self) -> bytes:
    if self._bytecode is not None:
      return self._bytecode
    output = io.BytesIO()
    self._module.operation.write_bytecode(file=output)
    return output.getvalue()

  def mlir_module(self, serialized: bool = True) -> Any:
    """Return string representation or MLIR Module."""
    if serialized:
      return str(self._module)
    return self._module

  def __str__(self) -> str:
    return str(self._module)

  def _compile(self):
    if self._compiled is None:
      backend = xb.get_backend()
      executable_devices = xc.DeviceList(tuple(backend.local_devices()))
      compile_options = xc.CompileOptions()
      self._compiled = backend.compile_and_load(
        str(self._module),
        executable_devices=executable_devices,
        compile_options=compile_options,
      )

  def call(self, *args, **kwargs) -> Any:
    """Executes the deserialized module with the given inputs."""
    self._compile()
    flat_args, _ = pytree.tree_flatten((args, kwargs))
    env = torchax.default_env()
    jax_args = []
    for arg in flat_args:
      if isinstance(arg, torch.Tensor):
        jax_args.append(env.t2j_copy(arg))
      elif isinstance(arg, np.ndarray):
        jax_args.append(jnp.asarray(arg))
      else:
        jax_args.append(arg)

    assert self._compiled is not None
    results = self._compiled.execute(jax_args)
    if isinstance(results, (list, tuple)) and len(results) == 1:
      return results[0]
    return results

  def __call__(self, *args, **kwargs) -> Any:
    return self.call(*args, **kwargs)


def deserialize_vhlo_artifact(
  bytecode_or_text: bytes | bytearray | str,
) -> jax.export.Exported | DeserializedStableHLO:
  """Deserializes a portable VHLO bytecode or MLIR text artifact.

  Args:
    bytecode_or_text: Serialized VHLO bytecode (bytes/bytearray) or MLIR
      text (str).

  Returns:
    An executable Exported or DeserializedStableHLO object with
    .mlir_module() and .call().

  Raises:
    ValueError: If the artifact is invalid, unsupported, or cannot be
      deserialized.
    TypeError: If input is not bytes, bytearray, or str.
  """
  if not isinstance(bytecode_or_text, (bytes, bytearray, str)):
    raise TypeError(
      f"Expected bytes, bytearray, or str, got {type(bytecode_or_text).__name__}"
    )

  if isinstance(bytecode_or_text, (bytes, bytearray)):
    raw_bytes = bytes(bytecode_or_text)
    # First, try deserializing as a jax.export.Exported flatbuffer
    try:
      return jax.export.deserialize(bytearray(raw_bytes))
    except Exception:
      pass

    # Next, try deserializing as a portable artifact bytecode
    try:
      with jax_mlir.make_ir_context() as context:
        try:
          # Newer JAX:
          module = _jax.mlir.deserialize_portable_artifact(raw_bytes, context=context)
        except TypeError:
          # Older JAX (CI environment):
          module_str = _jax.mlir.deserialize_portable_artifact(raw_bytes)

          context.allow_unregistered_dialects = True
          module = ir.Module.parse(module_str, context=context)
        return DeserializedStableHLO(module, bytecode=raw_bytes)
    except Exception as e:
      # If binary deserialization failed, check if it's utf-8 encoded text
      try:
        text = raw_bytes.decode("utf-8")
        return deserialize_vhlo_artifact(text)
      except Exception:
        raise ValueError(f"Failed to deserialize VHLO artifact bytecode: {e}") from e

  elif isinstance(bytecode_or_text, str):
    try:
      stablehlo_dialect.register_stablehlo_passes()
      with jax_mlir.make_ir_context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(bytecode_or_text, context=context)
        # If the text contains VHLO ops, legalize to StableHLO
        if "vhlo." in bytecode_or_text:
          current_ver = stablehlo_dialect.get_current_version()
          pipeline = f"builtin.module(vhlo-to-version{{target={current_ver}}},vhlo-legalize-to-stablehlo)"
          pass_manager = pm.PassManager.parse(pipeline, context=context)
          pass_manager.run(module.operation)
        return DeserializedStableHLO(module, text=bytecode_or_text)
    except Exception as e:
      raise ValueError(f"Failed to deserialize VHLO artifact text: {e}") from e


def exported_program_to_stablehlo(
  exported_program,
  target_version: str | None = None,
  output_format: str = "stablehlo",
):
  """Replacement for torch_xla.stablehlo.exported_program_to_stablehlo.

  Convert a program exported via torch.export to StableHLO or VHLO.

  This supports dynamic dimension sizes and generates explicit checks for
  dynamo guards in the IR using shape_assertion custom_call ops.

  Args:
    exported_program: ExportedProgram from torch.export.export.
    target_version: Optional VHLO target version (e.g. '1.0.0', '0.19.0').
    output_format: Output format: 'stablehlo', 'bytecode', 'text', 'vhlo',
      'mlir_text'.

  Returns:
    (weights, artifact): weights pytree and the exported artifact in requested
    format.
  """
  weights, func = exported_program_to_jax(exported_program)
  jax_avals = extract_avals(exported_program)
  jax_export = jax.export.export(jax.jit(func))(weights, (jax_avals,))

  fmt = output_format.lower()

  if target_version is not None:
    validate_target_version(target_version)
    target = target_version.strip()

    if fmt == "stablehlo":
      vhlo_bytecode = legalize_stablehlo_to_vhlo(
        jax_export.mlir_module(),
        target,
        output_format="bytecode",
      )
      updated_export = dataclasses.replace(
        jax_export, mlir_module_serialized=vhlo_bytecode
      )
      return weights, updated_export
    elif fmt in ("bytecode", "vhlo", "portable_artifact", "vhlo_bytecode"):
      vhlo_bytecode = legalize_stablehlo_to_vhlo(
        jax_export.mlir_module(),
        target,
        output_format="bytecode",
      )
      return weights, vhlo_bytecode
    elif fmt in ("text", "mlir_text", "vhlo_text", "mlir"):
      vhlo_text = legalize_stablehlo_to_vhlo(
        jax_export.mlir_module(), target, output_format="text"
      )
      return weights, vhlo_text
    else:
      raise ValueError(
        f"Unsupported output_format: '{output_format}'. Supported formats: "
        "'stablehlo', 'bytecode', 'vhlo', 'portable_artifact', 'vhlo_bytecode', 'text', 'mlir_text', 'vhlo_text', 'mlir'."
      )

  # target_version is None
  if fmt == "stablehlo":
    return weights, jax_export
  elif fmt in ("bytecode", "vhlo_bytecode", "portable_artifact", "vhlo"):
    vhlo_bytecode = legalize_stablehlo_to_vhlo(
      jax_export.mlir_module(), None, output_format="bytecode"
    )
    return weights, vhlo_bytecode
  elif fmt in ("text", "mlir_text", "mlir"):
    return weights, str(jax_export.mlir_module())
  elif fmt in ("vhlo_text",):
    vhlo_text = legalize_stablehlo_to_vhlo(
      jax_export.mlir_module(), None, output_format="text"
    )
    return weights, vhlo_text
  else:
    raise ValueError(
      f"Unsupported output_format: '{output_format}'. Supported formats: "
      "'stablehlo', 'bytecode', 'vhlo_bytecode', 'portable_artifact', 'vhlo', 'text', 'mlir_text', 'mlir', 'vhlo_text'."
    )
