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
import torchax as tx
import torchax.export
import torchax.interop
import torchax.train
import optax
from torch.testing._internal.common_utils import TestCase


class SimpleModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.layer1 = torch.nn.Linear(10, 20)
    self.output = torch.nn.Linear(20, 5)

  def forward(self, x):
    x = self.layer1(x)
    x = torch.nn.functional.relu(x)
    return self.output(x)


class TrainTest(TestCase):

  def setUp(self):
    torch.manual_seed(0)
    torchax.enable_accuracy_mode()

  def test_scan_module(self):
    x = torch.arange(300).reshape(3, 100).to(torch.float32)
    layers = [
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100),
    ]
    # repetitively applies the linear
    result = x
    for layer in layers:
      result = layer(result)

    model = tx.train.ScannedModule(layers)

    with torchax.default_env():
      x = x.to('jax')
      model.to('jax')
      result2 = model(x)
      torch.testing.assert_allclose(result, result2.to('cpu'))

  def test_train_step_can_run(self):
    import optax
    with torchax.default_env():
      model = torch.nn.Linear(100, 100)
      model.to('jax')
      weights = model.state_dict()
      x = torch.randn(2, 100).to('jax')
      y = torch.tensor([1, 2]).to('jax')

      def model_fn(weight, buffers, args):
        return torch.func.functional_call(model, weight, args)

      loss_fn = torch.nn.CrossEntropyLoss()

      optimizer = optax.adam(0.01)
      opt_state = tx.interop.call_jax(optimizer.init, weights)

      step = tx.train.make_train_step(model_fn, loss_fn, optimizer)
      print(step(weights, {}, opt_state, x, y))

  def test_freeze_layers(self):
    with torchax.default_env():
      model = SimpleModel().to("jax")

      jittable_mod = torchax.interop.JittableModule(model)
      initial_params = {k: v.clone() for k, v in jittable_mod.params.items()}
      initial_buffers = jittable_mod.buffers

      def model_fn(weights, buffers, args):
        return jittable_mod.functional_call("forward", weights, buffers, args)

      def loss_fn(predictions, targets):
        return torch.mean((predictions - targets)**2)

      optimizer = optax.sgd(0.01)

      # --- 1. Test with frozen layers ---
      # Freeze parameters in the first layer, similar to freeze_example.py
      for param in model.layer1.parameters():
          param.requires_grad = False

      train_step_frozen = torchax.train.make_train_step(
          model_fn, loss_fn, optimizer, frozen_params_filter=lambda n, v: not v.requires_grad)

      jitted_frozen_step = torchax.interop.jax_jit(train_step_frozen)

      inputs = torch.randn(32, 10, device="jax")
      labels = torch.randn(32, 5, device="jax")

      # The optimizer should only be initialized with trainable parameters.
      trainable_params = {k: v for k, v in initial_params.items() if v.requires_grad}
      opt_state_frozen = torchax.interop.call_jax(optimizer.init, trainable_params)

      # Note: The 'weights' passed to the step function still contain all parameters.
      _, updated_params_frozen, _ = jitted_frozen_step(initial_params,
                                                       initial_buffers,
                                                       opt_state_frozen, inputs,
                                                       labels)

      # Assertions for frozen run
      self.assertTrue(
          torch.allclose(initial_params["layer1.weight"],
                         updated_params_frozen["layer1.weight"]))
      self.assertFalse(
          torch.allclose(initial_params["output.weight"],
                         updated_params_frozen["output.weight"]))

      # --- 2. Test without freezing (control) ---
      # Re-enable gradients for the control test
      for param in model.layer1.parameters():
          param.requires_grad = True

      train_step_normal = torchax.train.make_train_step(model_fn, loss_fn, optimizer)
      jitted_normal_step = torchax.interop.jax_jit(train_step_normal)
      opt_state_normal = torchax.interop.call_jax(optimizer.init, jittable_mod.params)

      _, updated_params_normal, _ = jitted_normal_step(initial_params,
                                                       initial_buffers,
                                                       opt_state_normal, inputs,
                                                       labels)

      # Assertions for normal run
      self.assertFalse(
          torch.allclose(initial_params["layer1.weight"],
                         updated_params_normal["layer1.weight"]))
      self.assertFalse(
          torch.allclose(initial_params["output.weight"],
                         updated_params_normal["output.weight"]))


if __name__ == '__main__':
  unittest.main()
