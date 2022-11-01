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

"""Stein variational gradient descent preconditioner."""

from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


__all__ = [
    "SVGD",
]


class SVGD(Optimizer):
    """Stein variational gradient descent preconditioner.

    References
    ----------
    .. [1] Q. Liu and D. Wang.
           "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm".
           In: Advances in Neural Information Processing Systems. 2016, pp. 2378-2386.
           URL: https://arxiv.org/abs/1608.04471

    Examples
    --------


    """

    # override
    def __init__(
        self,
        params: "Union[Iterable[Tensor], Iterable[Dict[str, Any]]]",
        num_particles: "int",
        stein_kernel: "Callable[[Tensor], Tuple[Tensor, Tensor]]",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        params:
            The parameters to precondition (already
            replicated `num_particles` times).
        num_particles:
            The number of particles.
        stein_kernel:
            The Stein kernel, i.e. a callable that receives as an argument
            the particles and returns the corresponding kernels and kernel
            gradients.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if num_particles < 1 or not float(num_particles).is_integer():
            raise ValueError(
                f"`num_particles` ({num_particles}) must be in the integer interval [1, inf)"
            )

        super().__init__(params, {})
        self.num_particles = int(num_particles)
        self.stein_kernel = stein_kernel

    # override
    @torch.no_grad()
    def step(self) -> "None":
        for group in self.param_groups:
            params = group["params"]

            # Extract particles
            particles = nn.utils.parameters_to_vector(params).reshape(
                self.num_particles, -1
            )

            # Extract particle gradients
            particle_grads = nn.utils.parameters_to_vector(
                (param.grad for param in params)
            ).reshape(self.num_particles, -1)

            # Compute kernels and kernel gradients
            kernels, kernel_grads = self.stein_kernel(particles)

            # Compute total gradient
            particle_grads = kernels @ particle_grads  # Attractive gradients
            particle_grads -= kernel_grads  # Repulsive gradients
            particle_grads /= self.num_particles
            particle_grads = particle_grads.flatten()

            # Inject particle gradients
            start_idx = 0
            for param in params:
                num_elements = param.numel()
                param.grad = particle_grads[
                    start_idx : start_idx + num_elements
                ].reshape_as(param)
                start_idx += num_elements
