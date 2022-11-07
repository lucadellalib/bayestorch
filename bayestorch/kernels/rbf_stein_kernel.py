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

"""Radial basis function Stein kernel."""

import math
from typing import Tuple

import torch
from torch import Tensor


__all__ = [
    "RBFSteinKernel",
]


class RBFSteinKernel:
    """Radial basis function kernel to use in Stein
    variational gradient descent.

    The bandwidth of the kernel is chosen from the particles
    using a simple heuristic as in reference [1].

    References
    ----------
    .. [1] Q. Liu and D. Wang.
           "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm".
           In: Advances in Neural Information Processing Systems. 2016, pp. 2378-2386.
           URL: https://arxiv.org/abs/1608.04471

    Examples
    --------
    >>> import torch
    >>>
    >>> from bayestorch.kernels import RBFSteinKernel
    >>>
    >>>
    >>> num_particles = 5
    >>> particle_size = 1000
    >>> kernel = RBFSteinKernel()
    >>> particles = torch.rand(num_particles, particle_size)
    >>> kernels, kernel_grads = kernel(particles)

    """

    @torch.no_grad()
    def __call__(self, particles: "Tensor") -> "Tuple[Tensor, Tensor]":
        """Compute the kernels and the kernel gradients.

        In the following, let `N` denote the number
        of particles and `D` the particle size.

        Parameters
        ----------
        particles:
            The particles, shape: ``[N, D]``.

        Returns
        -------
            - The kernels, shape: ``[N, N]``;
            - the kernel gradients with respect
              to the particles, shape: ``[N, D]``.

        """
        num_particles = particles.shape[0]
        return self._forward(particles, num_particles)

    @staticmethod
    @torch.jit.script
    def _forward(particles: "Tensor", num_particles: "int") -> "Tuple[Tensor, Tensor]":
        deltas = torch.cdist(particles, particles)
        squared_deltas = deltas**2
        bandwidth = squared_deltas.median() / math.log(num_particles)
        log_kernels = -squared_deltas / bandwidth
        kernels = log_kernels.exp()
        kernel_grads = (
            (2 / bandwidth) * (kernels.sum(dim=-1).diag() - kernels) @ particles
        )
        return kernels, kernel_grads

    def __repr__(self) -> "str":
        return f"{type(self).__name__}()"
