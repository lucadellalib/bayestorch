# Adapted from:
# https://github.com/pytorch/examples/blob/9aad148615b7519eadfa1a60356116a50561f192/regression/main.py

# Changes to the code are kept to a minimum to facilitate the comparison with the original example

#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F

import math
from bayestorch.distributions import get_log_scale_normal
from bayestorch.nn import ParticlePosteriorModule
from bayestorch.optim import SVGD

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


def rbf_kernel(x1, x2):
    deltas = torch.cdist(x1, x2)
    squared_deltas = deltas**2
    bandwidth = (
        squared_deltas.detach().median()
        / math.log(min(x1.shape[0], x2.shape[0]))
    )
    log_kernels = -squared_deltas / bandwidth
    kernels = log_kernels.exp()
    return kernels


# Number of particles
num_particles = 10

# Log prior weight
log_prior_weight = 1e-1

# Define model
fc = torch.nn.Linear(W_target.size(0), 1)

# Prior arguments (WITHOUT gradient tracking)
prior_builder, prior_kwargs = get_log_scale_normal(fc.parameters(), 0.0, -1.0)

# Bayesian model
fc = ParticlePosteriorModule(fc, prior_builder, prior_kwargs, num_particles)

# SVGD preconditioner
preconditioner = SVGD(fc.parameters(include_all=False), rbf_kernel, num_particles)

for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    fc.zero_grad()

    # Forward pass
    #output = F.smooth_l1_loss(fc(batch_x), batch_y)
    #loss = output.item()
    output, log_prior = fc(batch_x, return_log_prior=True)
    loss = F.smooth_l1_loss(output, batch_y, reduction="sum") - log_prior_weight * log_prior

    # Backward pass
    loss.backward()
    loss = loss.item()

    # SVGD step
    preconditioner.step()

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
    torch.stack([replica.weight.view(-1) for replica in fc.replicas]).mean(dim=0),
    torch.stack([replica.bias for replica in fc.replicas]).mean(dim=0),
))
print('==> Learned function standard deviation:\t' + poly_desc(
    torch.stack([replica.weight.view(-1) for replica in fc.replicas]).std(dim=0),
    torch.stack([replica.bias for replica in fc.replicas]).std(dim=0),
))
print('==> Actual function:                    \t' + poly_desc(W_target.view(-1), b_target))
