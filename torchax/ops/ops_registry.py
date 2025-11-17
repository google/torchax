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

import dataclasses
import logging

from torchax.types import JaxCallable, TorchCallable


@dataclasses.dataclass
class Operator:
  torch_op: TorchCallable
  func: TorchCallable | JaxCallable
  is_jax_function: bool
  is_user_defined: bool
  needs_env: bool
  is_view_op: bool


all_aten_ops: dict[TorchCallable, Operator] = {}
all_torch_functions: dict[TorchCallable, Operator] = {}


def register_torch_dispatch_op(
  aten_op,
  impl_callable,
  is_jax_function=True,
  is_user_defined=False,
  needs_env=False,
  is_view_op=False,
):
  op = Operator(
    aten_op,
    impl_callable,
    is_jax_function=is_jax_function,
    is_user_defined=is_user_defined,
    needs_env=needs_env,
    is_view_op=is_view_op,
  )
  if aten_op in all_aten_ops:
    logging.warning(f"Duplicate op registration for {aten_op}")
  all_aten_ops[aten_op] = op
  return impl_callable


def register_torch_function_op(
  torch_func,
  impl_callable,
  is_jax_function=True,
  is_user_defined=False,
  needs_env=False,
  is_view_op=False,
):
  op = Operator(
    torch_func,
    impl_callable,
    is_jax_function=is_jax_function,
    is_user_defined=is_user_defined,
    needs_env=needs_env,
    is_view_op=is_view_op,
  )
  all_torch_functions[torch_func] = op
  return impl_callable
