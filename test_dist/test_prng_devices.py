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
from jax.sharding import NamedSharding, PartitionSpec

import torchax
from torchax import interop


class PRNGKeyDevicePlacementTest(unittest.TestCase):
  """The threaded PRNG key must follow the args' devices across calls."""

  def setUp(self):
    torchax.enable_globally()

  def tearDown(self):
    torchax.disable_globally()

  def _make_inputs(self):
    """Return a mesh over all devices, plus the same tensor sharded on it and on one device."""
    mesh = jax.make_mesh((len(jax.devices()),), ("x",))
    x = torch.ones(len(jax.devices()), 4, device="jax")
    sharded = x.apply_jax(jax.device_put, NamedSharding(mesh, PartitionSpec("x")))
    single = x.apply_jax(jax.device_put, jax.devices()[0])
    return mesh, sharded, single

  def test_jax_jit_sharded_then_single_device(self):
    _, sharded, single = self._make_inputs()
    fn = interop.jax_jit(lambda a: (a * 2).sum())

    fn(sharded)
    fn(single)
    fn(sharded)

  def test_jax_value_and_grad_sharded_then_single_device(self):
    _, sharded, single = self._make_inputs()
    fn = interop.jax_value_and_grad(lambda a: (a * 2).sum())

    fn(sharded)
    fn(single)
    fn(sharded)

  def test_jax_shard_map_after_single_device(self):
    mesh, sharded, single = self._make_inputs()
    spec = PartitionSpec("x")
    jitted_fn = interop.jax_jit(lambda a: a * 2)
    sharded_fn = interop.jax_shard_map(
      lambda a: a * 2, {"mesh": mesh, "in_specs": (spec,), "out_specs": spec}
    )

    jitted_fn(single)
    sharded_fn(sharded)


if __name__ == "__main__":
  unittest.main()
