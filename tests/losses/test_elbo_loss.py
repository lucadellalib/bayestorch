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

"""Test evidence lower bound (a.k.a. variational free energy) loss."""

import pytest
import torch

from bayestorch.losses import ELBOLoss


def test_elbo_loss() -> "None":
    num_mc_samples = 5
    num_train_batches = 100
    batch_size = 256
    log_likelihoods = torch.rand(num_mc_samples, batch_size)
    kl_divs = torch.rand(num_mc_samples)
    criterion = ELBOLoss(reduction="none")
    loss = criterion(log_likelihoods, kl_divs, num_train_batches)
    print(f"Number of Monte Carlo samples: {num_mc_samples}")
    print(f"Number of train batches: {num_train_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Log likelihoods shape: {log_likelihoods.shape}")
    print(f"Kullback-Leibler divergences shape: {kl_divs.shape}")
    print(f"Loss shape: {loss.shape}")


if __name__ == "__main__":
    pytest.main([__file__])
