---
title: JAX Foundations - Thinking in JAX
date: 2026-04-05 09:00
categories: [ML]
tags: [JAX, JIT, Autodiff, Pytree, Benchmark]
author: James Huang
---

JAX is really about writing **pure numerical functions**, then transforming those functions with tools like `jit`, `grad`, and `vmap`.
So instead of writing separate code for "normal execution", "batched execution", or "differentiable execution", we often write one clean function and let JAX transform it.

This post is my weekly learning note on that idea.
I also wanted this post to be more than a summary, so for each core concept I ran a small local experiment and recorded the result.

## Experiment Setup

All experiments below were run on:

- JAX `0.4.30`
- device: `CpuDevice(id=0)`
- platform: `macOS-15.6-arm64`

The exact numbers will vary on another machine, especially on GPU or TPU.
Still, the experiments are useful because they make the behavior of `jit`, `grad`, `vmap`, explicit randomness, and state handling much more concrete.

For timing experiments, I used `jax.block_until_ready(...)` so the measurement reflects real execution instead of just dispatch time.

# 1. JAX Is About Transforming Functions

The main thing that clicked for me this week is that JAX is easiest to understand if I stop thinking of it as just a tensor library.
The more useful mental model is:

- write pure functions
- compile them with `jit`
- differentiate them with `grad`
- batch them with `vmap`
- structure parameters and state as pytrees

That is why so many JAX tutorials keep emphasizing "thinking in JAX."
The syntax is familiar, but the programming model is much more compiler-driven and transformation-driven than regular Python.

# 2. Pure Functions and Immutability

One of the first ideas in JAX is that transformations work best on **pure functions**.
Inputs go in, outputs come out, and we avoid hidden side effects and in-place mutation.

That is also why JAX array updates look different from NumPy.
Instead of mutating an array in place, we create a new array:

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
y = x.at[0].set(10.0)

print("x:", x)
print("y:", y)
```

### Result

```text
x: [1.0, 2.0, 3.0]
y: [10.0, 2.0, 3.0]
```

The important point is that `x` stays unchanged.
That may feel less convenient than mutation at first, but it fits the JAX model much better.
Pure functions are easier to trace, compile, and differentiate.

# 3. `jax.jit` and Why Warmed Execution Matters

`jax.jit` takes a Python function and compiles it for efficient execution on JAX backends.
The interesting part is not only that it can be faster, but that it changes how we think about execution.

The function is first **traced**, then compiled based on input structure such as shape and dtype.
That means the first call and later calls can behave very differently.

To make that visible, I benchmarked a function with many elementwise operations:

```python
import time
import jax
import jax.numpy as jnp

def bench(fn, loops=20, warmup=1):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(loops):
        start = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)

x = jax.random.normal(jax.random.key(0), (4096, 1024), dtype=jnp.float32)

def heavy_elementwise(x):
    y = x
    for _ in range(8):
        y = jnp.tanh(y * 1.1 + 0.3) + 0.01 * y**2 - 0.1 * jnp.sin(y)
    return y.mean(axis=1)

heavy_jit = jax.jit(heavy_elementwise)

eager_ms = bench(lambda: heavy_elementwise(x))

start = time.perf_counter()
jax.block_until_ready(heavy_jit(x))
jit_first_ms = (time.perf_counter() - start) * 1000

jit_ms = bench(lambda: heavy_jit(x))
```

### Result

```text
eager avg: 410.956 ms
jit first call (compile + run): 381.309 ms
jit warmed avg: 145.621 ms
speedup after warmup: 2.82x
```

The most important takeaway is not the exact number.
It is the shape of the result:

- the first `jit` call includes compilation work
- warmed `jit` calls are much faster than eager execution on this workload
- performance discussion in JAX usually has to separate compile time from steady-state execution

This experiment made `jit` feel much less magical to me.
It is not just a "make fast" button; it changes the execution model.

## Why Can't We Just JIT Everything?

After seeing a benchmark like the one above, it is very tempting to ask:
why not just wrap every function in `jax.jit()` and call it a day?

The short answer is that `jit` only works well when JAX can trace the function into a stable computation graph.
That usually means:

- array-oriented numerical work
- stable shapes and dtypes
- little or no Python-side branching based on runtime values
- enough repeated calls to amortize compile cost

When those assumptions break, `jit` can either fail outright or become slower than eager execution.

## When `jit` Fails: Python Control Flow on Runtime Values

One of the most common failures happens when Python control flow depends on the **value** of a traced input.

```python
import jax

def f(x):
    if x > 0:
        return x
    else:
        return 2 * x

jax.jit(f)(10)
```

This raises a `TracerBoolConversionError`.
The reason is that inside `jit`, `x` is traced abstractly.
JAX knows things like shape and dtype, but at trace time it does not have the concrete runtime value needed for a Python `if`.

The same issue shows up with Python loops:

```python
def g(x, n):
    i = 0
    while i < n:
        i += 1
    return x + i

jax.jit(g)(10, 20)
```

This also fails for the same reason:
the Python `while` depends on the runtime value of `n`, but tracing happens before JAX knows that value concretely.

So a good rule of thumb is:

- Python `if`, `while`, and `for` are fine when their behavior depends on static Python values
- they become problematic when they depend on traced JAX values

## What `jit` Can See

Inside `jit`, traced values can affect compilation through their **static structure**, not arbitrary runtime value.
That usually means:

- shape
- dtype
- pytree structure

But not:

- whether `x > 0`
- whether `n < 20`
- how many times a Python loop should run based on traced data

This is one of the biggest mental shifts in JAX.
We still write Python syntax, but the compiled program is not "executing Python normally."

## Correct Pattern 1: Rewrite Value-Based Branching

If possible, rewrite the function to avoid Python control flow on traced values.
For elementwise cases, this can often be expressed directly with array operations.

```python
import jax
import jax.numpy as jnp

def f_rewritten(x):
    return jnp.where(x > 0, x, 2 * x)

print(jax.jit(f_rewritten)(10))
print(jax.jit(f_rewritten)(-3))
```

This works because `jnp.where(...)` is part of the traced computation, instead of asking Python to choose a branch.

For more structured branching, JAX also provides control-flow primitives such as `jax.lax.cond`.

## Correct Pattern 2: JIT Only the Expensive Inner Part

Sometimes the outer Python loop is driven by runtime logic that is awkward to rewrite, but the heavy numerical work inside the loop is still worth compiling.
In that case, we can JIT only the hot inner function:

```python
import jax

@jax.jit
def loop_body(prev_i):
    return prev_i + 1

def g_inner_jitted(x, n):
    i = 0
    while i < n:
        i = loop_body(i)
    return x + i

print(g_inner_jitted(10, 20))
```

This pattern is often more practical than trying to force the whole outer function into `jit`.
It is a good reminder that JAX performance is not all-or-nothing.
We can compile the expensive numerical kernel and leave the awkward Python control flow outside.

## Correct Pattern 3: Mark Some Arguments as Static

If a function really does need Python control flow based on an argument, and that argument only takes a small set of values, we can mark it as static.

```python
import jax

def f(x):
    if x > 0:
        return x
    else:
        return 2 * x

f_jit_static = jax.jit(f, static_argnums=0)
print(f_jit_static(10))
```

We can also do the same thing by name:

```python
from functools import partial
import jax

@partial(jax.jit, static_argnames=["n"])
def g_jit_decorated(x, n):
    i = 0
    while i < n:
        i += 1
    return x + i

print(g_jit_decorated(10, 20))
```

This works, but it comes with an important tradeoff:
JAX now treats that argument as part of the compilation key.
If the static argument changes often, JAX may recompile repeatedly.

So static arguments are a good idea only when:

- the set of values is small
- recompilation cost is acceptable
- the branching really belongs in Python

## JIT and Caching

The first call to a jitted function pays the compile cost.
Later calls can reuse the cached compiled program.
That reuse is exactly why `jit` becomes worthwhile for repeated workloads.

But caching only helps when the compiled function identity and relevant input signature stay stable.
This is why these two patterns are dangerous:

```python
from functools import partial
import jax

def unjitted_loop_body(prev_i):
    return prev_i + 1

def bad_partial_loop(x, n):
    i = 0
    while i < n:
        i = jax.jit(partial(unjitted_loop_body))(i)
    return x + i

def bad_lambda_loop(x, n):
    i = 0
    while i < n:
        i = jax.jit(lambda x: unjitted_loop_body(x))(i)
    return x + i
```

Each iteration creates a new function object, which can defeat caching and trigger repeated compilation.

The safer version is:

```python
import jax

def unjitted_loop_body(prev_i):
    return prev_i + 1

jitted_loop_body = jax.jit(unjitted_loop_body)

def good_cached_loop(x, n):
    i = 0
    while i < n:
        i = jitted_loop_body(i)
    return x + i
```

This is a very practical rule:
define jitted functions once, outside loops and temporary scopes, so JAX can reuse the cached compiled artifact.

## So When Should We Use `jit`?

After reading the docs and trying a few examples, my current rule of thumb is:

Use `jit` when:

- the function does meaningful numerical work
- the same function will be called many times
- input shapes are relatively stable
- most of the work is array computation rather than Python logic

Be careful with `jit` when:

- the function is tiny and called only once
- Python control flow depends on runtime tensor values
- argument values change in a way that forces recompilation
- the function is recreated repeatedly inside loops or closures

Do not think of `jit` as a decorator to sprinkle everywhere.
It is better to think of it as a compiler boundary.
The real question is not "can I decorate this function?"
The better question is "is this function a stable numerical kernel that is worth compiling?"

# 4. `grad` as a Program Transformation

`jax.grad` also becomes clearer when treated as a function transformation.
It takes a scalar-valued function and returns another function that computes its gradient.

I tested that with a tiny linear regression loss:

```python
import jax
import jax.numpy as jnp

X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
y_true = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
w = jnp.array([0.1, -0.2], dtype=jnp.float32)

def loss_fn(w):
    pred = X @ w
    return jnp.mean((pred - y_true) ** 2)

grad_auto = jax.grad(loss_fn)(w)
grad_manual = (2.0 / X.shape[0]) * (X.T @ ((X @ w) - y_true))

print("autodiff grad:", grad_auto)
print("manual grad:", grad_manual)
print("max abs diff:", jnp.max(jnp.abs(grad_auto - grad_manual)))
```

### Result

```text
autodiff grad: [-18.200000762939453, -23.200000762939453]
manual grad:   [-18.200000762939453, -23.200000762939453]
max abs diff:  0.0
```

This is simple, but it is exactly why `grad` feels powerful.
We write the mathematical function once, and JAX gives us another function representing its derivative.

# 5. `vmap` Removes a Python Loop

One of my favorite ideas in JAX 101 was `vmap`.
Instead of rewriting a function manually for batch inputs, we can keep a clean single-example function and ask JAX to vectorize it.

Here I compared three versions:

- a Python loop over the batch
- `vmap`
- `jit(vmap(...))`

```python
import time
import jax
import jax.numpy as jnp

def bench(fn, loops=5, warmup=1):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(loops):
        start = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)

w = jax.random.normal(jax.random.key(1), (256,), dtype=jnp.float32)
xs = jax.random.normal(jax.random.key(2), (512, 256), dtype=jnp.float32)

def predict(w, x):
    h = jnp.tanh(x + w)
    h = jnp.sin(h * 0.5) + h**2
    return jnp.sum(h)

def predict_loop(w, xs):
    return jnp.stack([predict(w, x) for x in xs])

predict_vmap = jax.vmap(predict, in_axes=(None, 0))
predict_vmap_jit = jax.jit(predict_vmap)

loop_out = predict_loop(w, xs)
vmap_out = predict_vmap(w, xs)
max_diff = jnp.max(jnp.abs(loop_out - vmap_out))

loop_ms = bench(lambda: predict_loop(w, xs))
vmap_ms = bench(lambda: predict_vmap(w, xs))

start = time.perf_counter()
jax.block_until_ready(predict_vmap_jit(w, xs))
vmap_jit_first_ms = (time.perf_counter() - start) * 1000

vmap_jit_ms = bench(lambda: predict_vmap_jit(w, xs))
```

### Result

```text
max abs diff loop vs vmap: 0.0
python loop avg: 255.480 ms
vmap avg: 12.747 ms
jit(vmap) first call: 82.292 ms
jit(vmap) warmed avg: 1.248 ms
```

This was probably the most dramatic experiment in the whole post.

- `vmap` matched the loop exactly
- `vmap` alone was about `20x` faster than the Python loop on my machine
- `jit(vmap(...))` pushed the warmed execution to about `205x` faster than the Python loop

This is a good example of JAX's design philosophy:
write a small function once, then transform it instead of rewriting it.

# 6. Pytrees Make Nested Structures First-Class

At first, pytrees looked like a minor implementation detail to me.
After reading more, I think they are one of the reasons JAX feels clean.

A pytree is a nested Python structure built from containers like lists, tuples, and dicts, with values stored at the leaves.
A single array is also a pytree, and in JAX an array counts as one leaf.

This matters because JAX transformations understand pytrees directly.
That means we can keep model parameters, gradients, optimizer state, or batches in natural nested containers instead of flattening everything by hand.

Here is a very small example showing how JAX counts leaves:

```python
import jax
import jax.numpy as jnp

example_trees = [
    [1, {"k1": 2, "k2": (3, 4)}, 5],
    {"a": 2, "b": (2, 3)},
    jnp.array([1, 2, 3]),
]

for pytree in example_trees:
    leaves = jax.tree.leaves(pytree)
    print(f"{repr(pytree):<35} -> {len(leaves)} leaves: {leaves}")
```

### Result

```text
[1, {'k1': 2, 'k2': (3, 4)}, 5] -> 5 leaves: [1, 2, 3, 4, 5]
{'a': 2, 'b': (2, 3)}           -> 3 leaves: [2, 2, 3]
Array([1, 2, 3], dtype=int32)   -> 1 leaves: [Array([1, 2, 3], dtype=int32)]
```

This example helped me see the rule more clearly:

- lists, tuples, and dicts define the tree structure
- the actual values at the bottom are the leaves
- an entire JAX array is treated as one leaf, not one leaf per element

The practical value of pytrees shows up most clearly with model parameters.
For example, if parameters are stored as a nested dict, gradients returned by `jax.grad(...)` will usually have exactly the same structure:

```python
import jax
import jax.numpy as jnp

params = {
    "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])},
    "layer2": {"w": jnp.array([3.0]), "b": jnp.array([1.0])},
}

def loss_fn(params):
    total = (
        jnp.sum(params["layer1"]["w"] ** 2)
        + jnp.sum(params["layer1"]["b"] ** 2)
        + jnp.sum(params["layer2"]["w"] ** 2)
        + jnp.sum(params["layer2"]["b"] ** 2)
    )
    return total

grads = jax.grad(loss_fn)(params)
print(grads)
```

### Result

```text
{
  'layer1': {'b': Array([1.], dtype=float32), 'w': Array([2., 4.], dtype=float32)},
  'layer2': {'b': Array([2.], dtype=float32), 'w': Array([6.], dtype=float32)}
}
```

The important point is not the exact numbers.
It is that `grads` has the same nested structure as `params`.
That makes parameter updates much easier, because JAX can apply transforms leaf-by-leaf while preserving the whole tree shape.

An even more useful pattern is that `tree.map(...)` can align **multiple pytrees at the same time**.
This is exactly what we want during parameter updates, because `params` and `grads` usually have the same tree structure.

```python
learning_rate = 0.1

updated_params = jax.tree.map(
    lambda p, g: p - learning_rate * g,
    params,
    grads,
)

print(updated_params)
```

### Result

```text
{
  'layer1': {'b': Array([0.4], dtype=float32), 'w': Array([0.8, 1.6], dtype=float32)},
  'layer2': {'b': Array([0.8], dtype=float32), 'w': Array([2.4], dtype=float32)}
}
```

This is where pytrees really start to feel powerful to me.
JAX walks both trees in lockstep, matches corresponding leaves, and applies the function leaf-by-leaf.

So instead of manually writing:

- update `layer1["w"]`
- update `layer1["b"]`
- update `layer2["w"]`
- update `layer2["b"]`

we just express the update rule once and let JAX apply it across the whole parameter tree.

We can also map a function over every leaf without manually traversing the nested structure:

```python
scaled_params = jax.tree.map(lambda x: x * 0.1, params)
print(scaled_params["layer1"]["w"])
```

### Result

```text
[0.1 0.2]
```

So for me, the main use of pytrees is this:
they let JAX work with real ML data structures in their natural nested form.

```python
import jax
import jax.numpy as jnp

params = {
    "layer1": {"w": jnp.ones((2, 3)), "b": jnp.zeros((3,))},
    "layer2": {"w": 2.0 * jnp.ones((3, 1)), "b": jnp.array([5.0])},
}

shifted = jax.tree_util.tree_map(lambda arr: arr + 1, params)
leaf_count = len(jax.tree_util.tree_leaves(params))

print("leaf count:", leaf_count)
print("original layer2.b:", params["layer2"]["b"])
print("shifted layer2.b:", shifted["layer2"]["b"])
```

### Result

```text
leaf count: 4
original layer2.b: [5.0]
shifted layer2.b: [6.0]
```

The point is not just that tree utilities exist.
It is that JAX transformations understand these nested structures naturally.
That becomes very important once model parameters stop fitting into one flat tensor.

# 7. Randomness Is Explicit

JAX handles randomness differently from libraries that hide RNG state globally.
Instead, JAX passes random state explicitly through keys.

I tested two cases:

- reuse the exact same key twice
- split the key and use the child keys separately

```python
import jax

key = jax.random.key(42)

same_a = jax.random.normal(key, (3,))
same_b = jax.random.normal(key, (3,))

k1, k2 = jax.random.split(key)
split_a = jax.random.normal(k1, (3,))
split_b = jax.random.normal(k2, (3,))

print("same key sample A:", same_a)
print("same key sample B:", same_b)
print("split key sample A:", split_a)
print("split key sample B:", split_b)
```

### Result

```text
same key sample A:  [0.18693547, -1.2806505, -1.5593132]
same key sample B:  [0.18693547, -1.2806505, -1.5593132]
split key sample A: [0.63330066, 0.9610921, 1.3625766]
split key sample B: [-0.5675502, 0.28439185, -0.9320608]
```

This experiment makes the rule very clear:

- reusing the same key gives the same randomness
- splitting keys gives independent streams

At first it feels more verbose than NumPy, but it is actually very consistent with the JAX philosophy of explicit state and reproducibility.

# 8. Hidden State vs Explicit State

JAX transformations do not play nicely with hidden mutable state.
That sounds abstract, so I tried a small experiment with a counter.

First, I used a class with hidden internal state.
Then I rewrote the same idea as a stateless function that takes state in and returns new state out.

```python
import jax

class Counter:
    def __init__(self):
        self.n = 0

    def count(self):
        self.n += 1
        return self.n

counter = Counter()
fast_count = jax.jit(counter.count)

print([int(fast_count()), int(fast_count()), int(fast_count())])
print("internal n:", counter.n)

def count_stateless(state):
    new_state = state + 1
    return new_state, new_state

fast_count_stateless = jax.jit(count_stateless)

state = 0
outputs = []
for _ in range(3):
    value, state = fast_count_stateless(state)
    outputs.append(int(value))

print(outputs)
print("final state:", state)
```

### Result

```text
hidden-state counter outputs: [1, 1, 1]
hidden-state internal n after jit calls: 1
stateless counter outputs: [1, 2, 3]
final explicit state: 3
```

This was one of the most useful experiments for me.
The hidden-state version is exactly the kind of Python code that feels natural at first, but it does not behave the way we want under `jit`.
Once state becomes explicit, the behavior becomes predictable and transformation-friendly.

# References

- [JAX 101](https://docs.jax.dev/en/latest/jax-101.html)
- [Quickstart: How to think in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)
