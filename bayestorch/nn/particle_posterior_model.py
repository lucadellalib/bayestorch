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
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn import Module, ModuleList

from bayestorch.nn.prior_model import PriorModel
from bayestorch.nn.utils import nested_apply


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
    >>> from bayestorch.nn import ParticlePosteriorModel
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
    >>> output = model(input)
    >>> outputs, log_priors = model(
    ...     input,
    ...     return_log_prior=True,
    ...     reduction="none",
    ... )

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
        model_parameters: "Optional[Iterable[Tensor]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        model:
            The model.
        prior_builder:
            The prior builder, i.e. a callable that receives keyword
            arguments and returns a prior with size (batch + event)
            equal to the length of the 1D tensor obtained by flattening
            and concatenating each tensor in `model_parameters`.
        prior_kwargs:
            The keyword arguments to pass to the prior builder.
            Tensor arguments are internally registered as parameters
            if their `requires_grad` attribute is True, as persistent
            buffers otherwise.
        num_particles:
            The number of particles.
        model_parameters:
            The model parameters over which the prior is defined.
            Useful to selectively define a prior over a restricted
            subset of modules/parameters.
            Default to ``model.parameters()``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        Warnings
        --------
        High memory usage is to be expected as `num_particles - 1`
        full copies of the model must be maintained internally.

        """
        if num_particles < 1 or not float(num_particles).is_integer():
            raise ValueError(
                f"`num_particles` ({num_particles}) must be in the integer interval [1, inf)"
            )

        super().__init__(model, prior_builder, prior_kwargs, model_parameters)
        self.num_particles = int(num_particles)

        # Replicate model (one replica for each particle)
        self.models = ModuleList(
            [model] + [copy.deepcopy(model) for _ in range(num_particles - 1)]
        )

        # Retrieve indices of the selected parameters
        model_parameter_idxes = []
        all_model_parameters = list(model.parameters())
        for parameter in self.model_parameters:
            for i, x in enumerate(all_model_parameters):
                if parameter is x:
                    model_parameter_idxes.append(i)
                    break

        for model in self.models:
            # Sample new particle
            new_particle = self.prior.sample()

            # Inject sampled particle
            start_idx = 0
            all_model_parameters = list(model.parameters())
            model_parameters = [
                all_model_parameters[idx] for idx in model_parameter_idxes
            ]
            for parameter in model_parameters:
                end_idx = start_idx + parameter.numel()
                new_parameter = new_particle[start_idx:end_idx].reshape_as(parameter)
                parameter.detach_().requires_grad_(False).copy_(
                    new_parameter
                ).requires_grad_()
                start_idx = end_idx

    # override
    def forward(
        self,
        *args: "Any",
        return_log_prior: "bool" = False,
        reduction: "str" = "mean",
        **kwargs: "Any",
    ) -> "Union[Any, Tuple[Any, Tensor]]":
        """Forward pass.

        In the following, let `N` denote the number of particles,
        `B = {B_1, ..., B_k}` the batch shape, and `O = {O_1, ..., O_m}`
        the shape of a leaf value of the underlying model output (can be
        a nested tensor).

        Parameters
        ----------
        args:
            The positional arguments to pass to the underlying model.
        return_log_prior:
            True to additionally return the log prior (usually
            required during training), False otherwise.
        reduction:
            The reduction to apply to the leaf values of the underlying
            model output and to the log prior (if `return_log_prior` is
            True) across particles. Must be one of the following:
            - "none": no reduction is applied;
            - "mean": the leaf values and the log prior are averaged
                      across particles.
        kwargs:
            The keyword arguments to pass to the underlying model.

        Returns
        -------
            - The output, shape of a leaf value: ``[N, *B, *O]``
              if `reduction` is "none" , ``[*B, *O]`` otherwise;
            - if `return_log_prior` is True, the log prior, shape:
              ``[N]`` if `reduction` is "none" , ``[]`` otherwise.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if reduction not in ["none", "mean"]:
            raise ValueError(
                f"`reduction` ({reduction}) must be one of {['none', 'mean']}"
            )

        # Forward pass
        outputs = [model(*args, **kwargs) for model in self.models]
        if reduction == "none":
            outputs = nested_apply(torch.stack, outputs)
        elif reduction == "mean":
            outputs = nested_apply(
                lambda inputs, dim: torch.mean(torch.stack(inputs, dim), dim), outputs
            )

        if not return_log_prior:
            return outputs

        # Extract particles
        particles = nn.utils.parameters_to_vector(self.models.parameters()).reshape(
            self.num_particles, -1
        )

        # Compute log prior
        log_priors = self.prior.log_prob(particles)
        if reduction == "mean":
            log_priors = log_priors.mean()

        return outputs, log_priors

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(model: {self.model}, "
            f"prior: {self.prior}, "
            f"num_particles: {self.num_particles}, "
            f"model_parameters: {sum(parameter.numel() for parameter in self.model_parameters)})"
        )
