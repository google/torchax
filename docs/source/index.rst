.. torchax documentation master file, created by Gemini.

###################
Welcome to TorchAX!
###################

**torchax** is a backend for PyTorch, allowing users to run
PyTorch on Google Cloud TPUs. **torchax** is also a library for providing
graph-level interoperability between PyTorch and JAX.

This means, with **torchax** you can:

* Run PyTorch code on TPUs with as little as 2 lines of code change.
* Call a JAX function from a PyTorch function, passing in ``jax.Array``s.
* Call a PyTorch function from a JAX function, passing in a ``torch.Tensor``s.
* Use JAX features such as ``jax.grad``, ``optax``, and ``GSPMD`` to train a PyTorch model.
* Use a PyTorch model as feature extractor and use it with a JAX model.

.. note::
   This documentation is currently a skeleton. Please help us by `contributing <https://github.com/pytorch/xla/tree/master/torchax>`_!

************
Getting Started
************

Installation
============

First install torch CPU:

.. code-block:: bash

   # On Linux.
   pip install torch --index-url https://download.pytorch.org/whl/cpu

   # Or on Mac.
   pip install torch

Then install JAX for the accelerator you want to use:

.. code-block:: bash

   # On Google Cloud TPU.
   pip install -U jax[tpu]

   # Or, on GPU machines.
   pip install -U jax[cuda12]

   # Or, on Linux CPU machines or Macs.
   pip install -U jax

Finally install torchax:

.. code-block:: bash

   pip install torchax


Run a Model
===========

To execute a model with ``torchax``, you need to enable torchax to capture PyTorch ops:

.. code-block:: python

    import torchax
    torchax.enable_globally()

Then, you can use a ``jax`` device:

.. code-block:: python

    import torch
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    inputs = torch.randn(3, 3, 28, 28, device='jax')
    m = MyModel().to('jax')
    res = m(inputs)
    print(type(res))  # outputs torchax.tensor.Tensor


*****************
Table of Contents
*****************

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   how_it_works
   core_concepts
   distributed
   interop
   examples

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
