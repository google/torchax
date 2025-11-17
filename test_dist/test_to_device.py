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
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import torchax


class ToDeviceTest(unittest.TestCase):
  def test_to_device_twice(self):
    env = torchax.default_env()
    mesh = jax.make_mesh((jax.device_count(),), ("axis",))
    with env:
      step1 = torch.ones(
        100,
        100,
      )
      step2 = torch.triu(step1, diagonal=1)
      step3 = step2.to(dtype=torch.bool, device="jax")
      step3.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
      print(step3.to("jax"))
      self.assertEqual(step3.device.type, "jax")


if __name__ == "__main__":
  unittest.main()
