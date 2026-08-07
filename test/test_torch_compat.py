# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import torchax
from torchax._torch_compat import get_aten_overload
from torchax.amp import CastPolicy, autocast_policy


def test_get_aten_overload_returns_existing_overload():
  assert get_aten_overload("prod", "default") is torch.ops.aten.prod.default


def test_get_aten_overload_returns_none_for_missing_overload():
  assert get_aten_overload("prod", "not_an_overload") is None


def test_available_named_tensor_overload_keeps_autocast_policy():
  named_prod = get_aten_overload("prod", "dim_Dimname")
  if named_prod is not None:
    assert autocast_policy[named_prod] is CastPolicy.FP32


def test_default_environment_loads_available_decompositions():
  assert torchax.default_env() is not None
