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

import torchax
from torch import nn
from torch.nn import functional as F
import torch

xla_env = torchax.enable_globally()


class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


m = MyModel()
m = m.to("jax")

# Execute this model using torch
inputs = torch.randn(3, 3, 28, 28, device="jax")

print(m(inputs))
print("---=====")

m_compiled = torchax.compile(m)

print(m_compiled(inputs))

print("---")
