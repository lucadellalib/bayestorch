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

"""Prior model."""

from typing import Any, Callable, Dict, Tuple, TypeVar

from torch import Tensor, nn
from torch.distributions import Distribution, Independent
from torch.nn import Module, Parameter

from bayestorch.models.utils import nested_stack


__all__ = [
    "PriorModel",
]


_T = TypeVar("_T", bound="PriorModel")


class PriorModel(Module):
    """Bayesian model that defines a prior over its parameters.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>>
    >>> from bayestorch.distributions import LogScaleNormal
    >>> from bayestorch.models import PriorModel
    >>>
    >>>
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> num_parameters = sum(parameter.numel() for parameter in model.parameters())
    >>> model = PriorModel(
    ...     model,
    ...     prior_builder=LogScaleNormal,
    ...     prior_kwargs={
    ...         "loc": torch.zeros(num_parameters),
    ...         "log_scale": torch.full((num_parameters,), -1.0),
    ...     },
    ... )
    >>> input = torch.rand(batch_size, in_features)
    >>> outputs, log_priors = model(input)

    """

    prior: "Distribution"
    """The prior distribution."""

    # override
    def __init__(
        self,
        model: "Module",
        prior_builder: "Callable[..., Distribution]",
        prior_kwargs: "Dict[str, Any]",
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

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__()
        self.model = model
        self.prior_builder = prior_builder
        self.prior_kwargs = prior_kwargs = {
            k: v for k, v in prior_kwargs.items()
        }  # Avoid side effects

        # Build prior
        self.prior = self._build_distribution("prior", prior_builder, prior_kwargs)

    # override
    def to(self, *args: "Any", **kwargs: "Any") -> "_T":
        self.model.to(*args, **kwargs)
        super().to(*args, **kwargs)

        # Rebuild prior with parameters on new device
        self.prior = self._build_distribution(
            "prior", self.prior_builder, self.prior_kwargs
        )

        return self

    # override
    def forward(self, *args: "Any", **kwargs: "Any") -> "Tuple[Any, Tensor]":
        """Forward pass.

        In the following, let `B = {B_1, ..., B_k}` denote the
        batch shape and `O = {O_1, ..., O_m}` the shape of a
        leaf value of the underlying model output (can be a
        nested tensor).

        Parameters
        ----------
        args:
            The positional arguments to pass to the underlying model.
        kwargs:
            The keyword arguments to pass to the underlying model.

        Returns
        -------
            - The outputs, shape of a leaf value: ``[1, *B, *O]``;
            - the log priors, shape: ``[1]``.

        """
        # Extract particle
        particle = nn.utils.parameters_to_vector(self.model.parameters())

        # Compute log priors
        log_priors = self.prior.log_prob(particle)[None]

        # Forward pass
        outputs = [self.model(*args, **kwargs)]

        return nested_stack(outputs), log_priors

    def _build_distribution(
        self,
        name: "str",
        distribution_builder: "Callable[..., Distribution]",
        distribution_kwargs: "Dict[str, Any]",
    ) -> "Distribution":
        # Extract particle
        # particle = model parameters flattened into a 1D vector
        particle = nn.utils.parameters_to_vector(self.model.parameters())

        # Build distribution
        for k, v in distribution_kwargs.items():
            key = f"{name}_{k}"
            if isinstance(v, Tensor):
                if key in self._parameters:
                    v = self._parameters[key]
                elif key in self._buffers:
                    v = self._buffers[key]
                else:
                    v = self._register_tensor(key, v.to(particle))
            distribution_kwargs[k] = v
        distribution = distribution_builder(**distribution_kwargs)

        # Adjust distribution shape
        batch_ndims = len(distribution.batch_shape)
        if batch_ndims > 0:
            distribution = Independent(distribution, batch_ndims)

        # Validate distribution event shape
        event_shape = distribution.event_shape
        if event_shape != (1,) and event_shape != particle.shape:
            raise ValueError(
                f"{name.capitalize()} event size ({event_shape.numel()}) must be "
                f"equal to the number of model parameters ({particle.numel()})"
            )

        return distribution

    def _register_tensor(self, name: "str", input: "Tensor") -> "Tensor":
        if input.requires_grad:
            input = Parameter(input)
            self.register_parameter(name, input)
        else:
            self.register_buffer(name, input)
        return input

    # override
    def __repr__(self) -> "str":
        return f"{type(self).__name__}(model: {self.model}, prior: {self.prior})"
