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

from typing import Any, Callable, Dict

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Independent
from torch.nn import Module


__all__ = [
    "PriorModel",
]


class PriorModel(Module):
    """Bayesian model that defines a prior over its parameters.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from torch.distributions import Normal
    >>>
    >>>
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> model = PriorModel(
    >>>     model,
    >>>     prior_builder=Normal,
    >>>     prior_kwargs={"loc": 0.0, "scale": 0.1},
    >>> )
    >>> input = torch.rand(batch_size, in_features)
    >>> output = model(input)

    """

    # override
    def __init__(
        self,
        model: "nn.Module",
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

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__()
        self.model = model
        self.prior_builder = prior_builder
        self.prior_kwargs = prior_kwargs

        # Extract particle
        # particle = model parameters flattened into a 1D vector
        particle = nn.utils.parameters_to_vector(model.parameters())

        # Build prior (WITHOUT gradient propagation)
        for k, v in prior_kwargs.items():
            buffer = None
            if isinstance(v, (int, float)):
                buffer = torch.full_like(particle, v)
            elif isinstance(v, Tensor):
                buffer = v.to(particle.device)
            if buffer is not None:
                self.register_buffer(f"prior_{k}", buffer)
                prior_kwargs[k] = buffer
        self._prior = prior_builder(**prior_kwargs)

        # Adjust prior event shape
        batch_shape = self._prior.batch_shape
        if len(batch_shape) > 0:
            self._prior = Independent(self._prior, len(batch_shape))
        event_shape = self._prior.event_shape
        if event_shape != particle.shape:
            raise ValueError(
                f"Prior event size ({event_shape.numel()}) must be equal to "
                f"the number of model parameters ({particle.shape.numel()}) "
            )

    # override
    def forward(self, *args: "Any", **kwargs: "Any") -> "Any":
        return self.model(*args, **kwargs)

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}" f"(model: {self.model}, " f"prior: {self._prior})"
        )
