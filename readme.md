# Rieoptax

![CI status](https://github.com/saitejautpala/rieoptax/workflows/tests/badge.svg)

## Introduction

Rieoptax is library for Riemannian Optimization in [JAX](https://github.com/google/jax).  The proposed library is mainly driven by the needs of efficient implementation of manifold-valued operations, optimization solvers and neural network layers readily compatible with GPU and even TPU processors.

### Blitz Intro to Riemannian Optimization


Riemannian optimization  considers the following problem

$$\min_{w \in \mathcal{M}} f(w)$$ where $f : \mathcal{M} \rightarrow \mathbb{R}$, and $\mathcal{M}$ denotes a Riemannian manifold. 
Instead of considering  as a constrained problem, Riemannian optimization views it as an unconstrained problem on the manifold space. Riemannian (stochastic) gradient descent generalizes the Euclidean gradient descent with intrinsic updates on manifold, i.e., $w_{t+1} = {\rm Exp}_{w_t}(- \eta_t \, {\rm grad} f(w_t))$, where ${\rm grad} f(w_t)$ is the Riemannian (stochastic) gradient, ${\rm Exp}_w(\cdot)$ is the Riemannian exponential map at $w$ and $\eta_t$ is the step size. 

### Quick start
 
Two main differences between Euclidean Optimization and Riemannian Optimization is Riemannian Gradient $\text{grad} f(w)$ and Riemannian Exponential map $\text{Exp}$. Main design goal of Rieoptax to handle above two things behind scenes and make it similar to standard optimization in [Optax](https://github.com/deepmind/optax)

![image](https://user-images.githubusercontent.com/73220310/194949472-6450893c-662d-4ca2-9180-d41d7c17778e.png)

For a complete example, see [notebooks](https://github.com/SaitejaUtpala/rieoptax/tree/master/notebooks) folder

## Overview

It consists of three module

1) [geometry](https://github.com/SaitejaUtpala/rieoptax/tree/master/rieoptax/geometry) : Implements several Riemannian manifolds of interest along with useful operations like Riemanian Exponential, Logarithmic and Euclidean gradient to Riemannian gradeint conversion rules

2) [mechanism](https://github.com/SaitejaUtpala/rieoptax/tree/master/rieoptax/mechanism) : Noise calibration for differentially private mechanism with manifold valued outputs

3) [optimizers](https://github.com/SaitejaUtpala/rieoptax/tree/master/rieoptax/optimizers) : Riemannian Optimization algorithms

## Installation

Currently installaion is done directly through github and it will soon be available through PyPI.

```
!pip install git+https://github.com/SaitejaUtpala/rieoptax.git
```


## Citing Rieoptax
Preprint Coming Soon! 
