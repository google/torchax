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

import collections

import jax
import optax
import torch

import torchax
from torchax import interop
from torchax.interop import torch_view

remat = torch_view(jax.remat)
mark_sharding = torch_view(jax.lax.with_sharding_constraint)


def make_train_step(model_fn, loss_fn, optax_optimizer, remat_policy=None):
  """Make a function that do one train step given model and loss.

  model_fn: a function representing the model's forward:
      i.e. has signature Callable[weights, buffers, args] -> result. Where,
      weights is a pytree of trainable parameters
      buffers is a pytree of non-trainable parameters / constants
      args is the input data loaded from the data set
      result is the return value of the model
  loss_fn: a function to compute loss.
      i.e. it has signature of Callable[result, label] -> loss
      where, result is what model_fn returned
        loss is loaded from the dataloader.
  optax_optimizer: the optimizer from optax library. for example, optax.adam
  remat_policy: One of jax.ad_checkpoint.checkpoint_policies, specifies how
      to do gradient checkpointing. If None, then it means checkpoint everything.
  """
  env = torchax.default_env()

  def loss(weights, buffers, args, label):  # inputs are XLATensor
    with env, jax.named_scope("compute_loss"):
      res = model_fn(weights, buffers, args)
      l = loss_fn(res, label)  # noqa: E741
      return l

  # loss = interop.gradient_checkpoint(loss, kwargs={'policy': remat_policy})
  grad_fn = interop.jax_value_and_grad(loss)

  def step(weights, buffers, opt_state, args, label):  # inputs are array
    with jax.named_scope("compute_gradient"):
      loss, gradient = grad_fn(weights, buffers, args, label)

    with jax.named_scope("optimizer_updates"):
      updates, opt_state = interop.call_jax(
        optax_optimizer.update, gradient, opt_state, weights
      )
      weights = interop.call_jax(optax.apply_updates, weights, updates)
    return loss, weights, opt_state

  # TODO: apply jax.jit so the user don't have to.
  return step


class Container:
  pass


class ScannedModule(torch.nn.Module):
  def __init__(self, module_list, checkpoint_policy=None):
    super().__init__()

    self.c = None
    assert module_list
    self.c = Container()
    self.c.one_mod = module_list[0]
    self.checkpoint_policy = checkpoint_policy

    weights = self._stack_layer_weights(module_list)
    self.layer_weights_keys = list(self.c.one_mod.state_dict().keys())
    self.params = torch.nn.ParameterDict(
      {self._param_name_new(k): v for k, v in weights.items()}
    )

  def _stack_layer_weights(self, module_list):
    # Create weights such that, for every [n, m] weights
    # becomes [k, n, m] where k is number of layer
    # i.e. stacking layer weights together
    temp = collections.defaultdict(list)
    for m in module_list:
      for k, v in m.state_dict().items():
        temp[k].append(v)
    res = {k: torch.stack(v) for k, v in temp.items()}
    return res

  def _param_name_new(self, old):
    return "___".join(old.split("."))

  def _param_name_old(self, new):
    return ".".join(new.split("___"))

  def forward(self, *args, **kwargs):
    assert not kwargs
    weights = {k: self.params[self._param_name_new(k)] for k in self.layer_weights_keys}
    scan = interop.torch_view(jax.lax.scan)

    def eval_one_layer(args, weight):
      # unpack args
      h, *rest = args
      newh = torch.func.functional_call(self.c.one_mod, weight, args)
      # next layer's input; and residual to be added to list
      return (newh, *rest), None

    _eval_one_layer = interop.gradient_checkpoint(
      eval_one_layer,
      kwargs={"policy": self.checkpoint_policy},
    )
    h, _ = scan(
      _eval_one_layer,
      args,
      weights,
    )
    return h[0]
