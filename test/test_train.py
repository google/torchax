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
from torch.testing._internal.common_utils import TestCase

import torchax as tx
import torchax.export
import torchax.train


class TrainTest(unittest.TestCase):
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
      x = x.to("jax")
      model.to("jax")
      result2 = model(x)
      torch.testing.assert_allclose(result, result2.to("cpu"))

  def test_train_step_can_run(self):
    import optax

    with torchax.default_env():
      model = torch.nn.Linear(100, 100)
      model.to("jax")
      weights = model.state_dict()
      x = torch.randn(2, 100).to("jax")
      y = torch.tensor([1, 2]).to("jax")

      def model_fn(weight, buffers, args):
        return torch.func.functional_call(model, weight, args)

      loss_fn = torch.nn.CrossEntropyLoss()

      optimizer = optax.adam(0.01)
      opt_state = tx.interop.call_jax(optimizer.init, weights)

      step = tx.train.make_train_step(model_fn, loss_fn, optimizer)
      print(step(weights, {}, opt_state, x, y))


if __name__ == "__main__":
  unittest.main()
