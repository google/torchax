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

import functools
import unittest

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
      outer_function,
      jittable_module.params,
      jittable_module.buffers,
      jax.random.PRNGKey(0),
      input_tensor,
    )

    assert torch.equal(output, expected_output)

  def test_forward_reads_rng_from_env(self):
    torchax.enable_globally()

    class PlusOne(torch.nn.Module):
      def forward(self, x):
        return x + 1

    jittable_module = interop.JittableModule(PlusOne().to("jax"))
    env = torchax.default_env()
    start_rng = env.prng_key

    with env:
      output = jittable_module(torch.ones(2, 2).to("jax"))

    self.assertTrue(torch.equal(output, torch.full((2, 2), 2.0).to("jax")))
    self.assertFalse(jax.numpy.array_equal(start_rng, env.prng_key))

  def test_functional_call_does_not_pass_rng_to_module(self):
    class PlusOne(torch.nn.Module):
      def forward(self, x):
        return x + 1

    jittable_module = interop.JittableModule(PlusOne())
    x = torch.randn(2, 2)
    expected = x + 1
    output = jittable_module.functional_call(
      "forward",
      jittable_module.params,
      jittable_module.buffers,
      jax.random.PRNGKey(0),
      x,
    )
    assert torch.equal(output, expected)

  def test_forward_controls_random_ops_from_env(self):
    torchax.enable_globally()

    class RandomOut(torch.nn.Module):
      def forward(self, x):
        return torch.randn_like(x)

    model = RandomOut().to("jax")
    jittable_module = interop.JittableModule(model)
    x = torch.ones(16, 16).to("jax")
    env = torchax.default_env()

    with env.override_property(prng=jax.random.PRNGKey(0)):
      same_rng_1 = jittable_module(x)
    with env.override_property(prng=jax.random.PRNGKey(0)):
      same_rng_2 = jittable_module(x)
    with env.override_property(prng=jax.random.PRNGKey(1)):
      different_rng = jittable_module(x)

    self.assertTrue(torch.equal(same_rng_1, same_rng_2))
    self.assertFalse(torch.equal(same_rng_1, different_rng))

  def test_forward_uses_supplied_env(self):
    torchax.enable_globally()

    class RandomOut(torch.nn.Module):
      def forward(self, x):
        return torch.randn_like(x)

    model = RandomOut().to("jax")
    module_env = torchax.tensor.Environment()
    jittable_module = interop.JittableModule(model, env=module_env)
    x = torch.ones(16, 16).to("jax")

    with module_env.override_property(prng=jax.random.PRNGKey(0)):
      same_rng_1 = jittable_module(x)
    with module_env.override_property(prng=jax.random.PRNGKey(0)):
      same_rng_2 = jittable_module(x)
    with module_env.override_property(prng=jax.random.PRNGKey(1)):
      different_rng = jittable_module(x)

    self.assertTrue(torch.equal(same_rng_1, same_rng_2))
    self.assertFalse(torch.equal(same_rng_1, different_rng))

  def test_with_rng_context_manager(self):
    torchax.enable_globally()

    class RandomOut(torch.nn.Module):
      def forward(self, x):
        return torch.randn_like(x)

    model = RandomOut().to("jax")
    jittable_module = interop.JittableModule(model)
    x = torch.ones(16, 16).to("jax")

    with jittable_module.with_rng(jax.random.PRNGKey(0)):
      same_rng_1 = jittable_module(x)
    with jittable_module.with_rng(jax.random.PRNGKey(0)):
      same_rng_2 = jittable_module(x)
    with jittable_module.with_rng(jax.random.PRNGKey(1)):
      different_rng = jittable_module(x)

    self.assertTrue(torch.equal(same_rng_1, same_rng_2))
    self.assertFalse(torch.equal(same_rng_1, different_rng))

  def test_functional_call_controls_random_ops(self):
    torchax.enable_globally()

    class RandomOut(torch.nn.Module):
      def forward(self, x):
        return torch.randn_like(x)

    model = RandomOut().to("jax")
    jittable_module = interop.JittableModule(model)
    x = torch.ones(16, 16).to("jax")

    jitted_functional = interop.jax_jit(
      functools.partial(jittable_module.functional_call, "forward")
    )

    same_rng_1 = jitted_functional(
      jittable_module.params,
      jittable_module.buffers,
      jax.random.PRNGKey(0),
      x,
    )
    same_rng_2 = jitted_functional(
      jittable_module.params,
      jittable_module.buffers,
      jax.random.PRNGKey(0),
      x,
    )
    different_rng = jitted_functional(
      jittable_module.params,
      jittable_module.buffers,
      jax.random.PRNGKey(1),
      x,
    )

    self.assertTrue(torch.equal(same_rng_1, same_rng_2))
    self.assertFalse(torch.equal(same_rng_1, different_rng))


if __name__ == "__main__":
  unittest.main()
