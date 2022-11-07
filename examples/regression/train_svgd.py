# Adapted from:
# https://github.com/pytorch/examples/blob/9aad148615b7519eadfa1a60356116a50561f192/regression/main.py#L1

# Changes to the code are kept to a minimum to facilitate the comparison with the original example

#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F

from bayestorch.distributions import LogScaleNormal
from bayestorch.kernels import RBFSteinKernel
from bayestorch.losses import NLUPLoss
from bayestorch.models import ParticlePosteriorModel
from bayestorch.preconditioners import SVGD

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


# Number of particles
num_particles = 10
# Log prior weight
log_prob_weight = 1e-1
# Define model
fc = torch.nn.Linear(W_target.size(0), 1)
num_parameters = sum(parameter.numel() for parameter in fc.parameters())
# Prior arguments (WITHOUT gradient propagation)
normal_prior_loc = torch.zeros(num_parameters)
normal_prior_log_scale = torch.full((num_parameters,), -1.0)
# Bayesian model
fc = ParticlePosteriorModel(
    fc,
    LogScaleNormal,
    {"loc": normal_prior_loc, "log_scale": normal_prior_log_scale},
    num_particles,
)
# SVGD kernel
kernel = RBFSteinKernel()
# SVGD preconditioner
preconditioner = SVGD(fc.parameters(), kernel, num_particles)
# Loss function
criterion = NLUPLoss()

for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    fc.zero_grad()

    # Forward pass
    #output = F.smooth_l1_loss(fc(batch_x), batch_y)
    #loss = output.item()
    outputs, log_probs = fc(batch_x)
    log_likelihoods = -F.smooth_l1_loss(
        outputs.flatten(0, 1), batch_y.repeat((num_particles, 1)), reduction="none",
    ).reshape(num_particles, -1)
    loss = criterion(log_likelihoods, log_probs, log_prob_weight)

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
    torch.stack([model.weight.view(-1) for model in fc.models]).mean(dim=0),
    torch.stack([model.bias for model in fc.models]).mean(dim=0),
))
print('==> Learned function standard deviation:\t' + poly_desc(
    torch.stack([model.weight.view(-1) for model in fc.models]).std(dim=0),
    torch.stack([model.bias for model in fc.models]).std(dim=0),
))
print('==> Actual function:                    \t' + poly_desc(W_target.view(-1), b_target))
