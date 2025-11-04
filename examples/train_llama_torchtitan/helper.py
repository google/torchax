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

import time
import jax
from jax.tree_util import tree_map
from jax.sharding import NamedSharding
from torchax import interop

P = jax.sharding.PartitionSpec


def compile_step_func(step, weights, buffers, opt_state, args, label, mesh):
  step, weights, buffers, opt_state, args, label = interop.jax_view(
      (step, weights, buffers, opt_state, args, label))
  wshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, weights)
  bshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, buffers)
  oshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, opt_state)
  print('Start compiling')
  start = time.perf_counter()
  lowered = jax.jit(
      step,
      donate_argnums=(0, 2),
      #in_shardings=shardings,
      out_shardings=(NamedSharding(mesh, P()), wshardings, oshardings),
  ).lower(weights, buffers, opt_state, args, label)
  #print(lowered.as_text())
  # import pdb; pdb.set_trace()
  print('program size:', len(lowered.as_text()) / 1e6, 'm chars')
  step_compiled = lowered.compile()
  end = time.perf_counter()
  print('End compiling', end - start)
  for co in step_compiled.cost_analysis():
    print('Cost analysis:', co)
    # print('Flops', co['flops'])
    # print('GB accessed', co['bytes accessed'] / 1e9)
  return interop.torch_view(step_compiled)
