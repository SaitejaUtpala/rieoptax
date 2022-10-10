# Rieoptax

![CI status](https://github.com/saitejautpala/rieoptax/workflows/tests/badge.svg)

## Introduction

Rieoptax is library for Riemannian Optimization in [JAX](https://github.com/google/jax).  The proposed library is mainly driven by the needs of efficient implementation of manifold-valued operations, optimization solvers and neural network layers readily compatible with GPU and even TPU processors.

### Blitz Intro to Riemannian Optimization


Riemannian optimization  considers the following problem

$$\min_{w \in \mathcal{M}} f(w)$$ where $f : \mathcal{M} \rightarrow \mathbb{R}$, and $\mathcal{M}$ denotes a Riemannian manifold. 
Instead of considering  as a constrained problem, Riemannian optimization views it as an unconstrained problem on the manifold space. Riemannian (stochastic) gradient descent generalizes the Euclidean gradient descent with intrinsic updates on manifold, i.e., $w_{t+1} = {\rm Exp}_{w_t}(- \eta_t \, {\rm grad} f(w_t))$, where ${\rm grad} f(w_t)$ is the Riemannian (stochastic) gradient, ${\rm Exp}_w(\cdot)$ is the Riemannian exponential map at $w$ and $\eta_t$ is the step size. 

### Quick start
 
Two main differences between Euclidean Optimization and Riemannian Optimization is Riemannian Gradient $\text{grad} f(w)$ and Riemannian Exponential map $$
 ```Python
from rieoptax.core import rgrad, ManifoldArray
```

## Overview

It consists of four module

1) [geometry](https://github.com/SaitejaUtpala/rieoptax/tree/master/rieoptax/geometry) : Implements several Riemannian manifolds of interest along with useful operations like Riemanian Exponential, Logarithmic and Euclidean gradient to Riemannian gradeint conversion rules

2) [mechnism](https://github.com/SaitejaUtpala/rieoptax/tree/master/rieoptax/mechanism) : Noise calibration for differentially private mechanism with manifold valued outputs

3) 

## Installation

Currently 

## Quickstart

Optax contains implementations of [many popular optimizers](https://optax.readthedocs.io/en/latest/api.html#Common-Optimizers) and
[loss functions](https://optax.readthedocs.io/en/latest/api.html#common-losses).
For example the following code snippet uses the Adam optimizer from `optax.adam`
and the mean squared error from `optax.l2_loss`. We initialize the optimizer
state using the `init` function and `params` of the model.

```python
optimizer = optax.adam(learning_rate)
# Obtain the `opt_state` that contains statistics for the optimizer.
params = {'w': jnp.ones((num_weights,))}
opt_state = optimizer.init(params)
```

To write the update loop we need a loss function that can be differentiated by
Jax (with `jax.grad` in this
example) to obtain the gradients.

```python
compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
grads = jax.grad(compute_loss)(params, xs, ys)
```

The gradients are then converted via `optimizer.update` to obtain the updates
that should be applied to the current params to obtain the new ones.
`optax.apply_updates` is a convinience utility to do this.

```python
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

You can continue the quick start in [the Optax quickstart notebook.](https://github.com/deepmind/optax/blob/master/examples/quick_start.ipynb)


## Components

We refer to the [docs](https://optax.readthedocs.io/en/latest/index.html)
for a detailed list of available Optax components. Here, we highlight
the main categories of buiilding blocks provided by Optax.

### Gradient Transformations ([transform.py](https://github.com/deepmind/optax/blob/master/optax/_src/transform.py))

One of the key building blocks of `optax` is a `GradientTransformation`.

Each transformation is defined two functions:

*   `state = init(params)`
*   `grads, state = update(grads, state, params=None)`

The `init` function initializes a (possibly empty) set of statistics (aka state)
and the `update` function transforms a candidate gradient given some statistics,
and (optionally) the current value of the parameters.

For example:

```python
tx = scale_by_rms()
state = tx.init(params)  # init stats
grads, state = tx.update(grads, state, params)  # transform & update stats.
```

### Composing Gradient Transformations ([combine.py](https://github.com/deepmind/optax/blob/master/optax/_src/combine.py))

The fact that transformations take candidate gradients as input and return
processed gradients as output (in contrast to returning the updated parameters)
is critical to allow to combine arbitrary transformations into a custom
optimiser / gradient processor, and also allows to combine transformations for
different gradients that operate on a shared set of variables.

For instance, `chain` combines them sequentially, and returns a
new `GradientTransformation` that applies several transformations in sequence.

For example:

```python
my_optimiser = chain(
    clip_by_global_norm(max_norm),
    scale_by_adam(eps=1e-4),
    scale(-learning_rate))
```

### Wrapping Gradient Transformations ([wrappers.py](https://github.com/deepmind/optax/blob/master/optax/_src/wrappers.py))

Optax also provides several wrappers that take a `GradientTransformation` as
input and return a new `GradientTransformation` that modifies the behaviour
of the inner transformation in a specific way.

For instance the `flatten` wrapper flattens gradients into a single large vector
before applying the inner GradientTransformation. The transformed updated are
then unflattened before being returned to the user. This can be used to reduce
the overhead of performing many calculations on lots of small variables,
at the cost of increasing memory usage.

For example:
```python
my_optimiser = flatten(adam(learning_rate))
```

Other examples of wrappers include accumulating gradients over multiple steps,
or applying the inner transformation only to specific parameters or at
specific steps.

### Schedules ([schedule.py](https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py))

Many popular transformations use time dependent components, e.g. to anneal
some hyper-parameter (e.g. the learning rate). Optax provides for this purpose
`schedules` that can be used to decay scalars as a function of a `step` count.

For example you may use a polynomial schedule (with `power=1`) to decay
a hyper-parameter linearly over a number of steps:

```python
schedule_fn = polynomial_schedule(
    init_value=1., end_value=0., power=1, transition_steps=5)

for step_count in range(6):
  print(schedule_fn(step_count))  # [1., 0.8, 0.6, 0.4, 0.2, 0.]
```

Schedules are used by certain gradient transformation, for instance:

```python
schedule_fn = polynomial_schedule(
    init_value=-learning_rate, end_value=0., power=1, transition_steps=5)
optimiser = chain(
    clip_by_global_norm(max_norm),
    scale_by_adam(eps=1e-4),
    scale_by_schedule(schedule_fn))
```

### Popular optimisers ([alias.py](https://github.com/deepmind/optax/blob/master/optax/_src/alias.py))

In addition to the low level building blocks we also provide aliases for popular
optimisers built using these components (e.g. RMSProp, Adam, AdamW, etc, ...).
These are all still instances of a `GradientTransformation`, and can therefore
be further combined with any of the individual building blocks.

For example:

```python
def adamw(learning_rate, b1, b2, eps, weight_decay):
  return chain(
      scale_by_adam(b1=b1, b2=b2, eps=eps),
      scale_and_decay(-learning_rate, weight_decay=weight_decay))
```

### Applying updates ([update.py](https://github.com/deepmind/optax/blob/master/optax/_src/update.py))

After transforming an update using a `GradientTransformation` or any custom
manipulation of the update, you will typically apply the update to a set
of parameters. This can be done trivially using `tree_map`. 

For convenience, we expose an `apply_updates` function to apply updates to
parameters. The function just adds the updates and the parameters together,
i.e. `tree_map(lambda p, u: p + u, params, updates)`.

```python
updates, state = tx.update(grads, state, params)  # transform & update stats.
new_params = optax.apply_updates(params, updates)  # update the parameters.
```

Note that separating gradient transformations from the parameter update is
critical to support composing sequence of transformations (e.g. `chain`), as
well as combine multiple updates to the same parameters (e.g. in multi-task
settings where different tasks need different sets of gradient transformations).

### Losses ([loss.py](https://github.com/deepmind/optax/blob/master/optax/_src/loss.py))

Optax provides a number of standard losses used in deep learning, such as
`l2_loss`, `softmax_cross_entropy`, `cosine_distance`, etc.

```python
loss = huber_loss(predictions, targets)
```

The losses accept batches as inputs, however they perform no reduction across
the batch dimension(s). This is trivial to do in JAX, for example:

```python
avg_loss = jnp.mean(huber_loss(predictions, targets))
sum_loss = jnp.sum(huber_loss(predictions, targets))
```

### Second Order ([second_order.py](https://github.com/deepmind/optax/blob/master/optax/_src/second_order.py))

Computing the Hessian or Fisher information matrices for neural networks is
typically intractable due to the quadratic memory requirements. Solving for the
diagonals of these matrices is often a better solution. The library offers
functions for computing these diagonals with sub-quadratic memory requirements.

### Stochastic gradient estimators ([stochastic_gradient_estimators.py](https://github.com/deepmind/optax/blob/master/optax/_src/stochastic_gradient_estimators.py))

Stochastic gradient estimators compute Monte Carlo estimates of gradients of
the expectation of a function under a distribution with respect to the
distribution's parameters.

Unbiased estimators, such as the score function estimator (REINFORCE),
pathwise estimator (reparameterization trick) or measure valued estimator,
are implemented: `score_function_jacobians`, `pathwise_jacobians` and `
measure_valued_jacobians`. Their applicability (both in terms of functions and
distributions) is discussed in their respective documentation.

Stochastic gradient estimators can be combined with common control variates for
variance reduction via `control_variates_jacobians`. For provided control
variates see `delta` and `moving_avg_baseline`.

The result of a gradient estimator or `control_variates_jacobians` contains the
Jacobians of the function with respect to the samples from the input
distribution. These can then be used to update distributional parameters, or
to assess gradient variance.

Example of how to use the `pathwise_jacobians` estimator:

```python
dist_params = [mean, log_scale]
function = lambda x: jnp.sum(x * weights)
jacobians = pathwise_jacobians(
      function, dist_params,
      utils.multi_normal, rng, num_samples)

mean_grads = jnp.mean(jacobians[0], axis=0)
log_scale_grads = jnp.mean(jacobians[1], axis=0)
grads = [mean_grads, log_scale_grads]
optim_update, optim_state = optim.update(grads, optim_state)
updated_dist_params = optax.apply_updates(dist_params, optim_update)
```

where `optim` is an optax optimizer.

## Citing Optax

Optax is part of the [DeepMind JAX Ecosystem], to cite Optax please use
the [DeepMind JAX Ecosystem citation].

[DeepMind JAX Ecosystem]: https://deepmind.com/blog/article/using-jax-to-accelerate-our-research "DeepMind JAX Ecosystem"
[DeepMind JAX Ecosystem citation]: https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt "Citation"
