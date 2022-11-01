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

import pytest
import torch

from bayestorch.preconditioners import SVGD


def test_svgd() -> "None":
    num_particles = 5
    particle_size = 1000
    preconditioner = SVGD()
    particles = torch.rand(num_particles, particle_size)
    # kernels, kernel_grads = kernel(particles)
    print(f"Number of particles: {num_particles}")
    print(f"Particle size: {particle_size}")
    # print(f"Kernels shape: {kernels.shape}")
    # print(f"Kernel gradients shape: {kernel_grads.shape}")


if __name__ == "__main__":
    pytest.main([__file__])
