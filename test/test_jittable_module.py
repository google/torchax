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
from torchax import interop
import torch


class MyAwesomeModel(torch.nn.Module):
  pass


class EvenMoreAwesomeModel(torch.nn.Module):
  pass


class JittableModuleTest(unittest.TestCase):
  def test_isinstance_works(self):
    # Export and check for composite operations
    model = MyAwesomeModel()
    jittable_module = interop.JittableModule(model)

    # jittable_module should remain an instance of MyAwesomeModel logicailly
    assert isinstance(jittable_module, MyAwesomeModel)

  def test_isinstance_does_not_mix(self):
    # Export and check for composite operations
    JittableAwesomeModel = interop.JittableModule(MyAwesomeModel())
    JittableMoreAwesomeModel = interop.JittableModule(EvenMoreAwesomeModel())

    # jittable_module should remain an instance of MyAwesomeModel logicailly
    assert isinstance(JittableAwesomeModel, MyAwesomeModel)
    assert not isinstance(JittableAwesomeModel, EvenMoreAwesomeModel)
    assert isinstance(JittableMoreAwesomeModel, EvenMoreAwesomeModel)
    assert not isinstance(JittableMoreAwesomeModel, MyAwesomeModel)

  def test_functional_call_callable(self):
    def outer_function(model, x):
      return x + 1

    model = MyAwesomeModel()
    jittable_module = interop.JittableModule(model)

    # Check if the jittable module can be called like a function
    input_tensor = torch.randn(1, 3, 224, 224)
    expected_output = input_tensor + 1

    output = jittable_module.functional_call(
      outer_function, jittable_module.params, jittable_module.buffers, input_tensor
    )

    assert torch.equal(output, expected_output)


if __name__ == "__main__":
  unittest.main()
