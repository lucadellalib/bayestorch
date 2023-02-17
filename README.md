# BayesTorch

[![Python version: 3.6 | 3.7 | 3.8 | 3.9 | 3.10](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/lucadellalib/bayestorch/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://github.com/PyCQA/isort)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![PyPI version](https://img.shields.io/pypi/v/bayestorch)
[![](https://pepy.tech/badge/bayestorch)](https://pypi.org/project/bayestorch/)

Welcome to `bayestorch`, a lightweight Bayesian deep learning library for fast prototyping based on
[PyTorch](https://pytorch.org). It provides the basic building blocks for the following
Bayesian inference algorithms:

- [Bayes by Backprop (BBB)](https://arxiv.org/abs/1505.05424)
- [Markov chain Monte Carlo (MCMC)](https://www.cs.toronto.edu/~radford/ftp/thesis.pdf)
- [Stein variational gradient descent (SVGD)](https://arxiv.org/abs/1608.04471)

---------------------------------------------------------------------------------------------------------

## 💡 Key features

- Low-code definition of Bayesian (or partially Bayesian) models
- Support for custom neural network layers
- Support for custom prior/posterior distributions
- Support for layer/parameter-wise prior/posterior distributions
- Support for composite prior/posterior distributions
- Highly modular object-oriented design
- User-friendly and easily extensible APIs
- Detailed API documentation

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

### Using Pip

First of all, install [Python 3.6 or later](https://www.python.org). Open a terminal and run:

```
pip install bayestorch
```

### From source

First of all, install [Python 3.6 or later](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a
terminal and run:

```
pip install -e .
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Here are a few code snippets showcasing some of the key features of the library.
For complete training loops, refer to `examples/mnist` and `examples/regression`.

### Bayesian model trainable via Bayes by Backprop

```python
from torch.nn import Linear

from bayestorch.distributions import (
    get_mixture_log_scale_normal,
    get_softplus_inv_scale_normal,
)
from bayestorch.nn import VariationalPosteriorModel


# Define model
model = Linear(5, 1)

# Define log scale normal mixture prior over the model parameters
prior_builder, prior_kwargs = get_mixture_log_scale_normal(
    model.parameters(),
    weights=[0.75, 0.25],
    locs=(0.0, 0.0),
    log_scales=(-1.0, -6.0)
)

# Define inverse softplus scale normal posterior over the model parameters
posterior_builder, posterior_kwargs = get_softplus_inv_scale_normal(
    model.parameters(), loc=0.0, softplus_inv_scale=-7.0, requires_grad=True,
)

# Define Bayesian model trainable via Bayes by Backprop
model = VariationalPosteriorModel(
    model, prior_builder, prior_kwargs, posterior_builder, posterior_kwargs
)
```

### Partially Bayesian model trainable via Bayes by Backprop

```python
from torch.nn import Linear

from bayestorch.distributions import (
    get_mixture_log_scale_normal,
    get_softplus_inv_scale_normal,
)
from bayestorch.nn import VariationalPosteriorModel


# Define model
model = Linear(5, 1)

# Define log scale normal mixture prior over `model.weight`
prior_builder, prior_kwargs = get_mixture_log_scale_normal(
    [model.weight],
    weights=[0.75, 0.25],
    locs=(0.0, 0.0),
    log_scales=(-1.0, -6.0)
)

# Define inverse softplus scale normal posterior over `model.weight`
posterior_builder, posterior_kwargs = get_softplus_inv_scale_normal(
    [model.weight], loc=0.0, softplus_inv_scale=-7.0, requires_grad=True,
)

# Define partially Bayesian model trainable via Bayes by Backprop
model = VariationalPosteriorModel(
    model, prior_builder, prior_kwargs,
    posterior_builder, posterior_kwargs, [model.weight],
)
```

### Composite prior

```python
from torch.distributions import Independent
from torch.nn import Linear

from bayestorch.distributions import (
    Concatenated,
    get_laplace,
    get_normal,
    get_softplus_inv_scale_normal,
)
from bayestorch.nn import VariationalPosteriorModel


# Define model
model = Linear(5, 1)

# Define normal prior over `model.weight`
weight_prior_builder, weight_prior_kwargs = get_normal(
    [model.weight],
    loc=0.0,
    scale=1.0,
    prefix="weight_",
)

# Define Laplace prior over `model.bias`
bias_prior_builder, bias_prior_kwargs = get_laplace(
    [model.bias],
    loc=0.0,
    scale=1.0,
    prefix="bias_",
)

# Define composite prior over the model parameters
prior_builder = (
    lambda **kwargs: Concatenated([
        Independent(weight_prior_builder(**kwargs), 1),
        Independent(bias_prior_builder(**kwargs), 1),
    ])
)
prior_kwargs = {**weight_prior_kwargs, **bias_prior_kwargs}

# Define inverse softplus scale normal posterior over the model parameters
posterior_builder, posterior_kwargs = get_softplus_inv_scale_normal(
    model.parameters(), loc=0.0, softplus_inv_scale=-7.0, requires_grad=True,
)

# Define Bayesian model trainable via Bayes by Backprop
model = VariationalPosteriorModel(
    model, prior_builder, prior_kwargs, posterior_builder, posterior_kwargs,
)
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
