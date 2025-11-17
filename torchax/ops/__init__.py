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


def all_aten_jax_ops():
  # to load the ops
  import torchax.ops.jaten  # type: ignore
  import torchax.ops.ops_registry  # type: ignore

  return {
    key: val.func
    for key, val in torchax.ops.ops_registry.all_aten_ops.items()
    if val.is_jax_function
  }
