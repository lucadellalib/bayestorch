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

"""Particle posterior module."""

import copy
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module, ModuleList, Parameter

from bayestorch.nn.prior_module import PriorModule
from bayestorch.nn.utils import nested_apply


__all__ = [
    "ParticlePosteriorModule",
]


class ParticlePosteriorModule(PriorModule):
    """Bayesian module that defines a prior and a particle-based
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
    >>> from bayestorch.nn import ParticlePosteriorModule
    >>>
    >>>
    >>> num_particles = 5
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> num_parameters = sum(parameter.numel() for parameter in model.parameters())
    >>> model = ParticlePosteriorModule(
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

    replicas: "ModuleList"
    """The module replicas (one for each particle)."""

    # override
    def __init__(
        self,
        module: "Module",
        prior_builder: "Callable[..., Distribution]",
        prior_kwargs: "Dict[str, Any]",
        num_particles: "int" = 10,
        module_parameters: "Optional[Iterable[Tensor]]" = None,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        module:
            The module.
        prior_builder:
            The prior builder, i.e. a callable that receives keyword
            arguments and returns a prior with size (batch + event)
            equal to the length of the 1D tensor obtained by flattening
            and concatenating each tensor in `module_parameters`.
        prior_kwargs:
            The keyword arguments to pass to the prior builder.
            Tensor arguments are internally registered as parameters
            if their `requires_grad` attribute is True, as persistent
            buffers otherwise.
        num_particles:
            The number of particles.
        module_parameters:
            The module parameters over which the prior is defined.
            Useful to selectively define a prior over a restricted
            subset of submodules/parameters.
            Default to ``module.parameters()``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        Warnings
        --------
        High memory usage is to be expected as `num_particles - 1`
        replicas of the module must be maintained internally.

        """
        if num_particles < 1 or not float(num_particles).is_integer():
            raise ValueError(
                f"`num_particles` ({num_particles}) must be in the integer interval [1, inf)"
            )

        super().__init__(module, prior_builder, prior_kwargs, module_parameters)
        self.num_particles = int(num_particles)

        # Replicate module (one replica for each particle)
        self.replicas = ModuleList(
            [module] + [copy.deepcopy(module) for _ in range(num_particles - 1)]
        )

        # Retrieve indices of the selected parameters
        self._module_parameter_idxes = []
        replica_parameters = list(module.parameters())
        for parameter in self.module_parameters:
            for i, x in enumerate(replica_parameters):
                if parameter is x:
                    self._module_parameter_idxes.append(i)
                    break

        for replica in self.replicas:
            # Sample new particle
            new_particle = self.prior.sample()

            # Inject sampled particle
            start_idx = 0
            replica_parameters = list(replica.parameters())
            module_parameters = [
                replica_parameters[idx] for idx in self._module_parameter_idxes
            ]
            for parameter in module_parameters:
                end_idx = start_idx + parameter.numel()
                new_parameter = new_particle[start_idx:end_idx].reshape_as(parameter)
                parameter.detach_().requires_grad_(False).copy_(
                    new_parameter
                ).requires_grad_()
                start_idx = end_idx

    # override
    def named_parameters(
        self,
        *args: "Any",
        include_all: "bool" = True,
        **kwargs: "Any",
    ) -> "Iterator[Tuple[str, Parameter]]":
        """Return the named parameters.

        Parameters
        ----------
        include_all:
            True to include all the named parameters,
            False to include only those over which the
            prior is defined.

        Returns
        -------
            The named parameters.

        """
        if include_all:
            return super(PriorModule, self).named_parameters(*args, **kwargs)
        named_parameters = dict(
            super(PriorModule, self).named_parameters(*args, **kwargs)
        )
        result = []
        for replica in self.replicas:
            replica_parameters = list(replica.parameters())
            for idx in self._module_parameter_idxes:
                for k, v in named_parameters.items():
                    if v is replica_parameters[idx]:
                        result.append((k, v))
                        break
        return result

    @property
    def particles(self) -> "Tensor":
        """Return the particles.

        In the following, let `N` denote the number of particles,
        and `D` the number of parameters over which the prior is
        defined.

        Returns
        -------
            The particles, shape: ``[N, D]``.

        """
        result = []
        for replica in self.replicas:
            replica_parameters = list(replica.parameters())
            module_parameters = [
                replica_parameters[idx] for idx in self._module_parameter_idxes
            ]
            for parameter in module_parameters:
                result.append(parameter.flatten())
        return torch.cat(result).reshape(self.num_particles, -1)

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
        the shape of a leaf value of the underlying module output (can be
        a nested tensor).

        Parameters
        ----------
        args:
            The positional arguments to pass to the underlying module.
        return_log_prior:
            True to additionally return the log prior (usually
            required during training), False otherwise.
        reduction:
            The reduction to apply to the leaf values of the underlying
            module output and to the log prior (if `return_log_prior` is
            True) across particles. Must be one of the following:
            - "none": no reduction is applied;
            - "mean": the leaf values and the log prior are averaged
                      across particles.
        kwargs:
            The keyword arguments to pass to the underlying module.

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
        outputs = [replica(*args, **kwargs) for replica in self.replicas]
        if reduction == "none":
            outputs = nested_apply(torch.stack, outputs)
        elif reduction == "mean":
            outputs = nested_apply(
                lambda inputs, dim: torch.mean(torch.stack(inputs, dim), dim), outputs
            )

        if not return_log_prior:
            return outputs

        # Extract particles
        particles = self.particles

        # Compute log prior
        log_priors = self.prior.log_prob(particles)
        if reduction == "mean":
            log_priors = log_priors.mean()

        return outputs, log_priors

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(module: {self.module}, "
            f"prior: {self.prior}, "
            f"num_particles: {self.num_particles}, "
            f"module_parameters: {sum(parameter.numel() for parameter in self.module_parameters)})"
        )
