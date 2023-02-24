#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test Stein variational gradient descent preconditioner."""

import math

import pytest
import torch
from torch import Tensor, nn

from bayestorch.optim import SVGD


def rbf_kernel(x1: "Tensor", x2: "Tensor") -> "Tensor":
    deltas = torch.cdist(x1, x2)
    squared_deltas = deltas**2
    bandwidth = squared_deltas.detach().median() / math.log(
        min(x1.shape[0], x2.shape[0])
    )
    log_kernels = -squared_deltas / bandwidth
    kernels = log_kernels.exp()
    return kernels


def test_svgd() -> "None":
    num_particles = 5
    batch_size = 10
    in_features = 4
    out_features = 2
    models = nn.ModuleList(
        [nn.Linear(in_features, out_features) for _ in range(num_particles)]
    )
    try:
        _ = SVGD(models.parameters(), num_particles=-1)
        _ = SVGD(models.parameters(), num_particles=0.5)
        _ = SVGD(models.parameters(), num_particles=3)
    except Exception:
        pass
    preconditioner = SVGD(models.parameters(), rbf_kernel, num_particles)
    input = torch.rand(batch_size, in_features)
    outputs = torch.cat([model(input) for model in models])
    loss = outputs.sum()
    loss.backward()
    grads_before = nn.utils.parameters_to_vector(
        (parameter.grad for parameter in models.parameters())
    )
    preconditioner.step()
    grads_after = nn.utils.parameters_to_vector(
        (parameter.grad for parameter in models.parameters())
    )
    print(preconditioner)
    print(f"Number of particles: {num_particles}")
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {(batch_size, in_features)}")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Gradients shape: {grads_before.shape}")
    assert not torch.allclose(grads_before, grads_after)


if __name__ == "__main__":
    pytest.main([__file__])
