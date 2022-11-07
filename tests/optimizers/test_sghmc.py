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

"""Test stochastic gradient Hamiltonian Monte Carlo optimizer."""

import pytest
import torch
from torch import nn

from bayestorch.optimizers import SGHMC


def test_sghmc() -> "None":
    batch_size = 10
    in_features = 4
    out_features = 2
    model = nn.Linear(in_features, out_features)
    optimizer = SGHMC(model.parameters())
    input = torch.rand(batch_size, in_features)
    output = model(input)
    loss = output.sum()
    loss.backward()
    params_before = nn.utils.parameters_to_vector(model.parameters())
    optimizer.step()
    params_after = nn.utils.parameters_to_vector(model.parameters())
    print(optimizer)
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {(batch_size, in_features)}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters shape: {params_before.shape}")
    assert not torch.allclose(params_before, params_after)


if __name__ == "__main__":
    pytest.main([__file__])
