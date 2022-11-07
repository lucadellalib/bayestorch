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

"""Particle posterior model."""

import copy
from typing import Any, Callable, Dict, Tuple

from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn import Module, ModuleList

from bayestorch.models.prior_model import PriorModel
from bayestorch.models.utils import nested_stack


__all__ = [
    "ParticlePosteriorModel",
]


class ParticlePosteriorModel(PriorModel):
    """Bayesian model that defines a prior and a particle-based
    posterior over its parameters.

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
    >>> from bayestorch.distributions import LogScaleNormal
    >>> from bayestorch.models import ParticlePosteriorModel
    >>>
    >>>
    >>> num_particles = 5
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> num_parameters = sum(parameter.numel() for parameter in model.parameters())
    >>> model = ParticlePosteriorModel(
    ...     model,
    ...     prior_builder=LogScaleNormal,
    ...     prior_kwargs={
    ...         "loc": torch.zeros(num_parameters),
    ...         "log_scale": torch.full((num_parameters,), -1.0),
    ...     },
    ...     num_particles=num_particles,
    ... )
    >>> input = torch.rand(batch_size, in_features)
    >>> outputs, log_priors = model(input)

    """

    models: "ModuleList"
    """The model replicas (one for each particle)."""

    # override
    def __init__(
        self,
        model: "Module",
        prior_builder: "Callable[..., Distribution]",
        prior_kwargs: "Dict[str, Any]",
        num_particles: "int" = 10,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        model:
            The model.
        prior_builder:
            The prior builder, i.e. a callable that receives
            keyword arguments and returns a prior.
        prior_kwargs:
            The keyword arguments to pass to the prior builder.
            Tensor arguments are internally registered as
            parameters if their `requires_grad` attribute
            is True, as persistent buffers otherwise.
        num_particles:
            The number of particles.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if num_particles < 1 or not float(num_particles).is_integer():
            raise ValueError(
                f"`num_particles` ({num_particles}) must be in the integer interval [1, inf)"
            )

        super().__init__(model, prior_builder, prior_kwargs)
        self.num_particles = num_particles

        # Replicate model (one replica for each particle)
        self.models = ModuleList(
            [model] + [copy.deepcopy(model) for _ in range(num_particles - 1)]
        )

        for model in self.models:
            # Sample new particle
            new_particle = self.prior.sample()

            # Inject sampled particle
            start_idx = 0
            for parameter in model.parameters():
                num_elements = parameter.numel()
                new_parameter = new_particle[
                    start_idx : start_idx + num_elements
                ].reshape_as(parameter)
                parameter.detach_().requires_grad_(False).copy_(
                    new_parameter
                ).requires_grad_(True)
                start_idx += num_elements

    # override
    def forward(self, *args: "Any", **kwargs: "Any") -> "Tuple[Any, Tensor]":
        """Forward pass.

        In the following, let `N` denote the number of particles,
        `B = {B_1, ..., B_k}` the batch shape, and `O = {O_1, ..., O_m}`
        the shape of a leaf value of the underlying model output (can be
        a nested tensor).

        Parameters
        ----------
        args:
            The positional arguments to pass to the underlying model.
        kwargs:
            The keyword arguments to pass to the underlying model.

        Returns
        -------
            - The outputs, shape of a leaf value: ``[N, *B, *O]``;
            - the log priors, shape: ``[N]``.

        """
        # Extract particles
        particles = nn.utils.parameters_to_vector(self.models.parameters()).reshape(
            self.num_particles, -1
        )

        # Compute log priors
        log_priors = self.prior.log_prob(particles)

        # Forward pass
        outputs = [model(*args, **kwargs) for model in self.models]

        return nested_stack(outputs), log_priors

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(model: {self.model}, "
            f"prior: {self.prior}, "
            f"num_particles: {self.num_particles})"
        )
