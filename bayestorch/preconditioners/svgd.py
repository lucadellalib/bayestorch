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
    >>> import torch
    >>> from torch import nn
    >>>
    >>> from bayestorch.kernels import RBFSteinKernel
    >>> from bayestorch.preconditioners import SVGD
    >>>
    >>>
    >>> num_particles = 5
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> models = nn.ModuleList(
    ...     [nn.Linear(in_features, out_features) for _ in range(num_particles)]
    ... )
    >>> kernel = RBFSteinKernel()
    >>> preconditioner = SVGD(models.parameters(), kernel, num_particles)
    >>> input = torch.rand(batch_size, in_features)
    >>> outputs = torch.cat([model(input) for model in models])
    >>> loss = outputs.sum()
    >>> loss.backward()
    >>> preconditioner.step()

    """

    # override
    def __init__(
        self,
        params: "Union[Iterable[Tensor], Iterable[Dict[str, Any]]]",
        kernel: "Callable[[Tensor], Tuple[Tensor, Tensor]]",
        num_particles: "int",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        params:
            The parameters to precondition. The total number of
            parameters must be a multiple of `num_particles`.
        kernel:
            The kernel, i.e. a callable that receives as an argument the
            particles and returns the corresponding kernels and kernel
            gradients.
        num_particles:
            The number of particles.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__(params, {"kernel": kernel, "num_particles": num_particles})
        self.kernel = kernel
        self.num_particles = num_particles = int(num_particles)

        # Check consistency between number of parameters
        # and number of particles for each group
        for group in self.param_groups:
            params = group["params"]
            num_particles = group["num_particles"]

            if num_particles < 1 or not float(num_particles).is_integer():
                raise ValueError(
                    f"`num_particles` ({num_particles}) must be in the integer interval [1, inf)"
                )

            # Extract particles
            particles = nn.utils.parameters_to_vector(params)

            if particles.numel() % num_particles != 0:
                raise ValueError(
                    f"Total number of parameters ({particles.numel()}) must "
                    f"be a multiple of `num_particles` ({num_particles})"
                )

    # override
    @torch.no_grad()
    def step(self) -> "None":
        for group in self.param_groups:
            params = group["params"]
            kernel = group["kernel"]
            num_particles = group["num_particles"]

            # Extract particles
            particles = nn.utils.parameters_to_vector(params).reshape(num_particles, -1)

            # Extract particle gradients
            particle_grads = []
            for param in params:
                grad = param.grad
                if grad is None:
                    raise RuntimeError("Gradient of some parameters is None")
                particle_grads.append(grad)
            particle_grads = nn.utils.parameters_to_vector(particle_grads).reshape(
                num_particles, -1
            )

            # Compute kernels and kernel gradients
            kernels, kernel_grads = kernel(particles)

            # Attractive gradients (already divided by `num_particles` => use `NLUPLoss` with reduction="mean")
            particle_grads = kernels @ particle_grads

            # Repulsive gradients
            particle_grads -= kernel_grads / num_particles

            # Flatten
            particle_grads = particle_grads.flatten()

            # Inject particle gradients
            start_idx = 0
            for param in params:
                num_elements = param.numel()
                param.grad = particle_grads[
                    start_idx : start_idx + num_elements
                ].reshape_as(param)
                start_idx += num_elements
