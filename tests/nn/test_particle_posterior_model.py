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

"""Test particle posterior model."""

import pytest
import torch
from torch import nn

from bayestorch.distributions import LogScaleNormal
from bayestorch.nn import ParticlePosteriorModel


def test_particle_posterior_model() -> "None":
    num_particles = 5
    batch_size = 10
    in_features = 4
    out_features = 2
    model = nn.Linear(in_features, out_features)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    model = ParticlePosteriorModel(
        model,
        prior_builder=LogScaleNormal,
        prior_kwargs={
            "loc": torch.zeros(num_parameters),
            "log_scale": torch.full((num_parameters,), -1.0),
        },
        num_particles=num_particles,
    )
    input = torch.rand(batch_size, in_features)
    output = model(input)
    outputs, log_priors = model(
        input,
        return_log_prior=True,
        reduction="none",
    )
    print(model)
    print(f"Number of particles: {num_particles}")
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {(batch_size, in_features)}")
    print(f"Output shape: {output.shape}")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Log priors shape: {log_priors.shape}")


if __name__ == "__main__":
    pytest.main([__file__])