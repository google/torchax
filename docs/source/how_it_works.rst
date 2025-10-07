.. _how_it_works:

############
How it Works
############

Tensor subclass and eager mode
==============================

The class ``torchax.tensor.Tensor`` is a ``torch.Tensor`` subclass
that overrides ``__torch_dispatch__``.

It roughly looks like this (with some details removed):

The complete class impl is at `tensor.py`_

_`tensor.py`:https://github.com/pytorch/xla/blob/master/torchax/torchax/tensor.py

.. code-block:: python

    class Tensor(torch.Tensor):

        @staticmethod
        def __new__(cls, elem):
            return torch.Tensor._make_wrapper_subclass(
                cls,
                shape,
                dtype=dtype,
                device='meta',
                requires_grad=False,
            )

        def __init__(self, elem: jax.Array):
            super().__init__()
            self._elem = elem

        __torch_function__ = torch._C._disabled_torch_function_impl

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            # here assumes ALL tensors in args / kwargs are
            # instances of Tensor
            args, kwargs = unwrap((args, kwargs))
            jax_func = some_registry[func]
            res = jax_func(*args, **kwargs)
            return wrap(res)

        def wrap(tree):
            # wrap jax.Array with Tensor
            return pytree.tree_map_only(
                jax.Array, Tensor, tree)

        def unwrap(tree):
            # get jax.Array out ofTensor
            return pytree.tree_map_only(
                Tensor, lambda x: x._elem, tree)


In other words, assuming that we have a function
that takes ``jax.Array`` as input and returns ``jax.Array``
but otherwise implement the same semantics
as a `ATen` op; then, using this tensor we would
be able to route the call to this jax function.

The `jaten.py`_ file defines some of those ops.

.. _`jaten.py`: https://github.com/pytorch/xla/blob/master/torchax/torchax/ops/jaten.py

Let's take ``aten::matmul`` as example:

.. code-block:: python
    @op(torch.ops.aten.matmul)
    def _aten_matmul(x, y):
      return jnp.matmul(x, y)

The ``@op`` decorator just puts this function into ``some_registry`` dictionary.

``_aten_add`` has same signature as ``torch.ops.aten.add`` but takes ``jax.Array`` as
input.

.. image:: torchax.png
  :align: center

Now, when ``torch.matmul`` is called; we will get the information and swap it to
a JAX version of that function.

In otherwords, because our tensor subclass implements all the operators that PyTorch
has, thus, a PyTorch model cannot tell a difference because it and regular CPU tensor.
( In other words, our subclass obeys the `Liskov Substitution Principle`_).

.. _Liskov Substitution Principle: https://en.wikipedia.org/wiki/Liskov_substitution_principle

Therefore, when a Pytorch Model (``torch.nn.Module``) receives our special tensor, it
will do math on it without knowing that we sneaked in a ``jax.Array``.

.. image:: image-trojan.png


**NOTE:** This mechanism of using a Python numerical library as a PyTorch backend
is not JAX specific. Here is an example here Apple MLX is used as a PyTorch backend
the same way: https://github.com/qihqi/tnt
