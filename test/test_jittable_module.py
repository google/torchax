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
import functools

import jax
import torch

import torchax
from torchax import interop


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

  def test_take_rng_requires_rng_kwarg(self):
    class PlusOne(torch.nn.Module):
      def forward(self, x):
        return x + 1

    jittable_module = interop.JittableModule(PlusOne(), take_rng=True)
    with self.assertRaisesRegex(TypeError, "requires a `rng` kwarg"):
      jittable_module(torch.ones(2, 2))

  def test_take_rng_removes_rng_before_module_call(self):
    class PlusOne(torch.nn.Module):
      def forward(self, x):
        return x + 1

    jittable_module = interop.JittableModule(PlusOne(), take_rng=True)
    x = torch.randn(2, 2)
    expected = x + 1
    output = jittable_module.functional_call(
      "forward", jittable_module.params, jittable_module.buffers, x, rng=object()
    )
    assert torch.equal(output, expected)

  def test_take_rng_controls_random_ops(self):
    torchax.enable_globally()

    class RandomOut(torch.nn.Module):
      def forward(self, x):
        return torch.randn_like(x)

    model = RandomOut().to("jax")
    jittable_module = interop.JittableModule(model, take_rng=True)
    x = torch.ones(16, 16).to("jax")

    same_rng_1 = jittable_module(x, rng=jax.random.PRNGKey(0))
    same_rng_2 = jittable_module(x, rng=jax.random.PRNGKey(0))
    different_rng = jittable_module(x, rng=jax.random.PRNGKey(1))

    self.assertTrue(torch.equal(same_rng_1, same_rng_2))
    self.assertFalse(torch.equal(same_rng_1, different_rng))

  def test_take_rng_controls_random_ops_for_jitted_functional_call(self):
    torchax.enable_globally()

    class RandomOut(torch.nn.Module):
      def forward(self, x):
        return torch.randn_like(x)

    model = RandomOut().to("jax")
    jittable_module = interop.JittableModule(model, take_rng=True)
    x = torch.ones(16, 16).to("jax")

    jitted_functional = interop.jax_jit(
      functools.partial(jittable_module.functional_call, "forward")
    )

    same_rng_1 = jitted_functional(
      jittable_module.params, jittable_module.buffers, x, rng=jax.random.PRNGKey(0)
    )
    same_rng_2 = jitted_functional(
      jittable_module.params, jittable_module.buffers, x, rng=jax.random.PRNGKey(0)
    )
    different_rng = jitted_functional(
      jittable_module.params, jittable_module.buffers, x, rng=jax.random.PRNGKey(1)
    )

    self.assertTrue(torch.equal(same_rng_1, same_rng_2))
    self.assertFalse(torch.equal(same_rng_1, different_rng))


if __name__ == "__main__":
  unittest.main()
