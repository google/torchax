## What is torchax

**torchax** is a PyTorch frontend for JAX. It gives JAX the ability
to author JAX programs using familiar PyTorch syntax. It also provides
JAX-Pytorch interoperability, meaning, one can mix JAX & Pytorch syntax
together when authoring ML programs, and run it in every hardware JAX can
run.

With **torchax**, you can:
* Call JAX functions from PyTorch, passing in `jax.Array`s.
* Call PyTorch functions from JAX, passing in `torch.Tensor`s.
* Use JAX features like `jax.grad`, `optax`, and `GSPMD` to train PyTorch
  models.
* Use a PyTorch model as a feature extractor with a JAX model.
* Run PyTorch code on hardwares where JAX is supported, such as Google TPUs,
  with minimal code changes.


## Installation

First install torch CPU:

``` bash
# On Linux.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or on Mac.
pip install torch
```

Then install JAX for the accelerator you want to use:

``` bash
# On Google Cloud TPU.
pip install -U jax[tpu]

# Or, on GPU machines.
pip install -U jax[cuda12]

# Or, on Linux CPU machines or Macs.
pip install -U jax
```

Finally install torchax:

``` bash
pip install torchax
```

You can also install from source if you prefer the lastest torchax:

``` bash
pip install git+https://github.com/google/torchax.git@main
```

Note that for now we don't automatically install `torch` or `jax`
when installing `torchax` because we want to expose the option
of picking a version to the users.
