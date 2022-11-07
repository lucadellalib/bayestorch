# Adapted from:
# https://github.com/pytorch/examples/blob/9aad148615b7519eadfa1a60356116a50561f192/regression/main.py#L1

# Changes to the code are kept to a minimum to facilitate the comparison with the original example

#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F

from bayestorch.distributions import LogScaleNormal, SoftplusInvScaleNormal
from bayestorch.losses import ELBOLoss
from bayestorch.models import VariationalPosteriorModel

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
num_parameters = sum(parameter.numel() for parameter in fc.parameters())
# Prior arguments (WITHOUT gradient propagation)
normal_prior_loc = torch.zeros(num_parameters)
normal_prior_log_scale = torch.full((num_parameters,), -1.0)
# Posterior arguments (WITH gradient propagation)
normal_posterior_loc = torch.zeros(num_parameters, requires_grad=True)
normal_posterior_softplus_inv_scale = torch.full((num_parameters,), -7.0, requires_grad=True)
# Bayesian model
fc = VariationalPosteriorModel(
    fc,
    LogScaleNormal,
    {"loc": normal_prior_loc, "log_scale": normal_prior_log_scale},
    SoftplusInvScaleNormal,
    {"loc": normal_posterior_loc, "softplus_inv_scale": normal_posterior_softplus_inv_scale},
)
# Loss function
criterion = ELBOLoss()

for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    fc.zero_grad()

    # Forward pass
    #output = F.smooth_l1_loss(fc(batch_x), batch_y)
    #loss = output.item()
    outputs, kl_divs = fc(batch_x, num_mc_samples=num_mc_samples)
    log_likelihoods = -F.smooth_l1_loss(
        outputs.flatten(0, 1), batch_y.repeat((num_mc_samples, 1)), reduction="none",
    ).reshape(num_mc_samples, -1)
    loss = criterion(log_likelihoods, kl_divs, kl_div_weight)

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
