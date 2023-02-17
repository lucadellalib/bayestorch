# Adapted from:
# https://github.com/pytorch/examples/blob/9aad148615b7519eadfa1a60356116a50561f192/regression/main.py

# Changes to the code are kept to a minimum to facilitate the comparison with the original example

#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F

from bayestorch.distributions import get_log_scale_normal, get_softplus_inv_scale_normal
from bayestorch.nn import VariationalPosteriorModel

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


# Number of Monte Carlo samples
num_mc_samples = 10

# Kullback-Leibler divergence weight
kl_div_weight = 1e-1

# Define model
fc = torch.nn.Linear(W_target.size(0), 1)

# Prior arguments (WITHOUT gradient tracking)
prior_builder, prior_kwargs = get_log_scale_normal(fc.parameters(), 0.0, -1.0)

# Posterior arguments (WITH gradient tracking)
posterior_builder, posterior_kwargs = get_softplus_inv_scale_normal(fc.parameters(), 0.0, -7.0, requires_grad=True)

# Bayesian model
fc = VariationalPosteriorModel(fc, prior_builder, prior_kwargs, posterior_builder, posterior_kwargs)

for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    fc.zero_grad()

    # Forward pass
    #output = F.smooth_l1_loss(fc(batch_x), batch_y)
    #loss = output.item()
    output, kl_div = fc(batch_x, num_mc_samples=num_mc_samples, return_kl_div=True)
    loss = F.smooth_l1_loss(output, batch_y, reduction="sum") + kl_div_weight * kl_div

    # Backward pass
    loss.backward()
    loss = loss.item()

    # Apply gradients
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad)

    # Stop criterion
    if loss < 1e2:
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
#print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
#print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
print('==> Learned function mean:              \t' + poly_desc(
    fc.posterior_loc[:-1],
    fc.posterior_loc[-1][None],
))
print('==> Learned function standard deviation:\t' + poly_desc(
    F.softplus(fc.posterior_softplus_inv_scale[:-1]),
    F.softplus(fc.posterior_softplus_inv_scale[-1][None]),
))
print('==> Actual function:                    \t' + poly_desc(W_target.view(-1), b_target))
