# Working with impure functions

**Author: Han Qi**
**Date: Jan 4, 2026**

Jax transforms usually requires one to pass in a 
"pure function". A pure functions is the one without 
side effects and whose output only depends on it's inputs.

This usually means:

* The function should not read from a global state (say, a global)
  and depend on it's value
* The function should not write out to a global state

However, in reality, we can still work with functions that violates
the above with few tricks. The key is to understand how `jax.jit`'s 
tracing works.

## Pure function and tracing:

First I'll copy-paste a snippet from [Jax sharp edges](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) below, with
few modifications:

The following code:
```python
import numpy as np
from jax import jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp

def impure_print_side_effect(x):
  print("HERE", x)
  print("Executing function")  # This is a side-effect
  return x

# The side-effects appear during the first run
print ("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print ("Second call: ", jit(impure_print_side_effect)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))
```

Yields:
```bash
HERE JitTracer<~float32[]>
Executing function
First call:  4.0
Second call:  5.0
HERE JitTracer<float32[1]>
Executing function
Third call, different type:  [5.]
```

There we can deduce the following:

1. Jax.jit calls your function the first time it's called with a particular kind of 
   inputs (usually means fixing shape and dtype, but that is considered implementation detail); and 
   caches the compiled artifact.
2. if Jax calls the compiled artifict directly, the original function is not called again (thus does not print in the second call).
3. The argument the function receives is the tracer, not the origial numerical input.


## Globals are burnt in

Consider the following snippet
```python

import numpy as np
from jax import jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp

g = 0.
def impure_uses_globals(x):
  return x + g

# JAX captures the value of the global during the first run
print ("First call: ", jit(impure_uses_globals)(4.))
g = 10.  # Update the global

# Subsequent runs may silently use the cached value of the globals
print ("Second call: ", jit(impure_uses_globals)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
# This will end up reading the latest value of the global
print ("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))
```
yield:
```
First call:  4.0
Second call:  5.0
Third call, different type:  [14.]
```

and 

```python
g = 0.
def impure_saves_global(x):
  global g
  g = x
  return x

# JAX runs once the transformed function with special Traced values for arguments
print ("First call: ", jit(impure_saves_global)(4.))
print ("Saved global: ", g)  # Saved global has an internal JAX value
```

```
First call:  4.0
Saved global:  JitTracer(~float32[])
```

Take aways from the above example:

1. Globals are hard-coded (burnt in) in the compiled artifact as a constant. So
   future value changes will not manifest.
2. Write out to globals are useless, because jax only calls the function on tracing
   so we'd be writing out the value of the `Tracer` instead.
 
 
## Dealing with functions that read from globals, and write to globals

From the above, a key aspect to understand is that for a `jax.jit`'d
function to take consideration of an input, it has to be passed as an input,
and, for it to express a change of some global state, it has to return it 
in return value.

With the above in mind, let's see how to transform a function that read / write
to globals to a pure function.


Consider the following function:

```python
global_a = 10
global_b = None


def impure_f(x, y):
  print("arguments", x, y)
  global global_b
  global_b = y + 2  # writes to global
  return global_a + x  # reads global


jitted_f = jax.jit(impure_f)

print("first call", jitted_f(10, 10))
global_a = 20
print("second call", jitted_f(10, 10))
print("global b after ", global_b)
```

Yields
```
first call 20
second call 20
global b after  JitTracer<~int32[]>
```

We can transform the `impure_f` into a pure function with
the following trick:

```python
def pure_f(x, y, a):  # extra arg for every global
  global global_a
  global global_b
  # Step 1: save old values of global_a and global_b
  old_global_a = global_a
  old_global_b = global_b

  # Step 2: modify global_a to be a
  # note this will be the Tracer object when jitting
  global_a = a
  res = impure_f(x, y)  # call the function as is

  # the above function will modify global_b
  modified_global_b = global_b
  
  # reset values
  global_a = old_global_a
  global_b = old_global_b

  return res, modified_global_b  # return extra args


jitted_pure_f = jax.jit(pure_f)


print("===")
global_a = 10
print("first call", jitted_pure_f(10, 10, global_a))
global_a = 20
print("second call", jitted_pure_f(10, 10, global_a))
print("global b after ", global_b)
```

Yield:
```
arguments JitTracer<~int32[]> JitTracer<~int32[]>
first call (Array(20, dtype=int32, weak_type=True), Array(12, dtype=int32, weak_type=True))
second call (Array(30, dtype=int32, weak_type=True), Array(12, dtype=int32, weak_type=True))
global b after  None
```

Note that the above function, despite of having `global` declarations and
writes to global, it is a pure function:
1. it's value depends on only inputs, here we added extra inputs to represent the dependency to `global_a`
2. writes to globals results to extra value (and thus does not leak `Tracer`).

Now the above is a pure function.

## Implication to random number seed, and class states

The above is the same trick detailed in this issue: https://github.com/google/torchax/issues/17#issuecomment-3445098772

We can treat the RNG seed as another global state that many Pytorch programs depends. 

Note that global states are not just top level globals, they can also appear as class attributes
when applying `jit` to a method. This [article from Jax documentation](https://docs.jax.dev/en/latest/stateful-computations.html) describes this case thoroughly, so we will not expand here. This is also
the same trick employed by [JittableModule](https://github.com/google/torchax/blob/79d339e5a7b9908672cb746a53b09ca23eee2b82/torchax/interop.py#L134)
