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

"""Test prior module."""

import pytest
import torch
from torch import nn

from bayestorch.distributions import LogScaleNormal
from bayestorch.nn import PriorModule


def test_prior_module() -> "None":
    batch_size = 10
    in_features = 4
    out_features = 2
    model = nn.Linear(in_features, out_features)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    model = PriorModule(
        model,
        prior_builder=LogScaleNormal,
        prior_kwargs={
            "loc": torch.zeros(num_parameters),
            "log_scale": torch.full((num_parameters,), -1.0),
        },
    ).to("cpu")
    input = torch.rand(batch_size, in_features)
    _ = model(input)
    output, log_prior = model(input, return_log_prior=True)
    print(model)
    print(dict(model.named_parameters()).keys())
    print(model.parameters())
    print(dict(model.named_parameters(include_all=False)).keys())
    print(model.parameters(include_all=False))
    state_dict = model.state_dict()
    model.load_state_dict(state_dict)
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {(batch_size, in_features)}")
    print(f"Output shape: {output.shape}")
    print(f"Log prior shape: {log_prior.shape}")


if __name__ == "__main__":
    pytest.main([__file__])
