# torchax: Running PyTorch on TPU via JAX

![](docs/docs/assets/logo.jpeg)

**torchax** is a backend for PyTorch that allows users to run
PyTorch programs on Google Cloud TPUs. It also provides graph-level
interoperability between PyTorch and JAX.

With **torchax**, you can:
* Run PyTorch code on TPUs with minimal code changes.
* Call JAX functions from PyTorch, passing in `jax.Array`s.
* Call PyTorch functions from JAX, passing in `torch.Tensor`s.
* Use JAX features like `jax.grad`, `optax`, and `GSPMD` to train PyTorch
  models.
* Use a PyTorch model as a feature extractor with a JAX model.

## Install

First, install the CPU version of PyTorch:

```bash
# On Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu

# On Mac
pip install torch
```

Next, install JAX for your desired accelerator:

```bash
# On Google Cloud TPU
pip install -U jax[tpu]

# On GPU machines
pip install -U jax[cuda12]

# On Linux CPU machines or Macs (see the note below)
pip install -U jax
```

Note: For Apple devices, you can install the [Metal version](https://developer.apple.com/metal/jax/) of JAX for
hardware acceleration.

Finally, install torchax:

```bash
# Install from PyPI
pip install torchax

# Or, install torchax from source.
pip install git+https://github.com/google/torchax
```

## Running a Model

To execute a model with torchax, start with any `torch.nn.Module`.
Here’s an example with a simple 2-layer model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


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

# Execute this model using torch.
inputs = torch.randn(3, 3, 28, 28)
print(m(inputs))
```

To execute this model with `torchax`, we need to enable torchax to capture PyTorch ops:

```python
import torchax
torchax.enable_globally()
```

Then, we can use a `jax` device:

```python
inputs = torch.randn(3, 3, 28, 28, device='jax')
m = MyModel().to('jax')
res = m(inputs)
print(type(res))  # outputs torchax.tensor.Tensor
print(res.jax()) # print the underlying Jax Array
```

`torchax.tensor.Tensor` is a `torch.Tensor` subclass that holds
a `jax.Array`. You can inspect that JAX array with `res.jax()`.

Although the code appears to be standard PyTorch, it's actually running on JAX.

## How It Works

torchax uses a `torch.Tensor` subclass, `torchax.tensor.Tensor`, which holds a
`jax.Array` and overrides the `__torch_dispatch__` method. When a PyTorch operation
is executed within the torchax environment (enabled by `torchax.enable_globally()`),
the implementation of that operation is swapped with its JAX equivalent.

When a model is instantiated, tensor constructors like `torch.rand` create
`torchax.tensor.Tensor` objects containing `jax.Arrays`. Subsequent operations
extract the `jax.Array`, call the corresponding JAX implementation, and wrap the
result back into a `torchax.tensor.Tensor`.

For more details, see the [How It Works](docs/docs/user_guide/how-it-works.md) and
[Ops Registry](docs/ops_registry.md) documentation.

### Executing with `jax.jit`

While torchax can run models in eager mode, `jax.jit` can be used for better performance.
`jax.jit` is a decorator that compiles a function that takes and returns `torch.Tensors`
into a faster, JAX-compiled version.

To use `jax.jit`, you first need a functional version of your model where parameters
are passed as inputs:

```python
def model_func(param, inputs):
  return torch.func.functional_call(m, param, inputs)
```

Here we use [torch.func.functional_call](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html)
from PyTorch to replace the model weights with `param` and then call the
model. This is roughly equivalent to:

```python
def model_func(param, inputs):
  m.load_state_dict(param)
  return m(*inputs)
```

Now, we can apply `jax_jit` on `module_func`:

```python
from torchax.interop import jax_jit

model_func_jitted = jax_jit(model_func)
print(model_func_jitted(new_state_dict, inputs))
```

See more examples at [eager_mode.py](examples/eager_mode.py) and the
[examples folder](examples/).

To ease the idiom of creating functional model and calling it with parameters,
we also created the `JittableModule` helper class. It lets us rewrite the
above as:

```python
from torchax.interop import JittableModule

m_jitted = JittableModule(m)
res = m_jitted(...)
```

The first time `m_jitted` is called, it will trigger `jax.jit` to compile the
compile for the given input shapes. Subsequent calls with the same input shapes
will be fast as the compilation is cached.

## Saving and Loading Checkpoints

You can save and load your training state using `torchax.save_checkpoint` and `torchax.load_checkpoint`.
The state can be a dictionary containing the model's weights, optimizer state, and any other relevant
information.

```python
import torchax
import torch
import optax

# Assume model, optimizer, and other states are defined
model = MyModel()
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(model.parameters())
weights = model.parameters()
buffers = model.buffers()
epoch = 10

state = {
    'weights': weights,
    'buffers': buffers,
    'opt_state': opt_state,
    'epoch': epoch,
}

# Save checkpoint
torchax.save_checkpoint(state, '/path/to/checkpoint.pt')

# Load checkpoint
loaded_state = torchax.load_checkpoint('/path/to/checkpoint.pt')

# Restore state
model.load_state_dict(loaded_state['weights'])
opt_state = loaded_state['opt_state']
epoch = loaded_state['epoch']
```

## Citation

```
@software{torchax,
  author = {Han Qi, Chun-nien Chan, Will Cromar, Manfei Bai, Kevin Gleanson},
  title = {torchax: PyTorch on TPU and JAX interoperability},
  url = {https://github.com/pytorch/xla/tree/master/torchax}
  version = {0.0.4},
  date = {2025-02-24},
}
```

## Maintainers & Contributors

This library is maintained by a team within Google Cloud. It has benefited from
many contributions from both inside and outside the team.

Thank you to recent contributors.

```
Han Qi (qihqi), PyTorch/XLA
Manfei Bai (manfeibai), PyTorch/XLA
Will Cromar (will-cromar), Meta
Milad Mohammadi (miladm), PyTorch/XLA
Siyuan Liu (lsy323), PyTorch/XLA
Bhavya Bahl (bhavya01), PyTorch/XLA
Pei Zhang (zpcore), PyTorch/XLA
Yifei Teng (tengyifei), PyTorch/XLA
Chunnien Chan (chunnienc), Google, ODML
Alban Desmaison (albanD), Meta, PyTorch
Simon Teo (simonteozw), Google (20%)
David Huang (dvhg), Google (20%)
Barni Seetharaman (barney-s), Google (20%)
Anish Karthik (anishfish2), Google (20%)
Yao Gu (guyao), Google (20%)
Yenkai Wang (yenkwang), Google (20%)
Greg Shikhman (commander), Google (20%)
Matin Akhlaghinia (matinehAkhlaghinia), Google (20%)
Tracy Chen (tracych477), Google (20%)
Matthias Guenther (mrguenther), Google (20%)
WenXin Dong (wenxindongwork), Google (20%)
Kevin Gleason (GleasonK), Google, StableHLO
Nupur Baghel (nupurbaghel), Google (20%)
Gwen Mittertreiner (gmittert), Google (20%)
Zeev Melumian (zmelumian), Lightricks
Vyom Sharma (vyom1611), Google (20%)
Shitong Wang (ShitongWang), Adobe
Rémi Doreau (ayshiff), Google (20%)
Lance Wang (wang2yn84), Google, CoreML
Hossein Sarshar (hosseinsarshar), Google (20%)
Daniel Vega-Myhre (danielvegamyhre), Google (20%)
Tianqi Fan (tqfan28), Google (20%)
Jim Lin (jimlinntu), Google (20%)
Fanhai Lu (FanhaiLu1), Google Cloud
DeWitt Clinton (dewitt), Google PyTorch
Aman Gupta (aman2930), Google (20%)
```

A special thank you to @albanD for the [initial inspiration](https://github.com/albanD/subclass_zoo/blob/main/new_device.py)
for torchax.
