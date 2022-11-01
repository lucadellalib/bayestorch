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

"""Variational posterior model."""

import copy
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, kl_divergence
from torch.nn import Parameter

from .prior_model import PriorModel
from .utils import nested_stack


__all__ = [
    "VariationalPosteriorModel",
]


_LOGGER = logging.getLogger(__name__)


class VariationalPosteriorModel(PriorModel):
    """Bayesian model that defines a prior and a variational
    posterior over its parameters.

    References
    ----------
    .. [1] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra.
           "Weight Uncertainty in Neural Networks".
           In: ICML. 2015, pp. 1613-1622.
           URL: https://arxiv.org/abs/1505.05424

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from torch.distributions import Normal
    >>>
    >>>
    >>> num_mc_samples = 5
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> model = VariationalPosteriorModel(
    >>>     model,
    >>>     prior_builder=Normal,
    >>>     prior_kwargs={"loc": 0.0, "scale": 0.1},
    >>>     posterior_builder=Normal,
    >>>     posterior_kwargs={"loc": 0.0, "scale": 0.3},
    >>> )
    >>> input = torch.rand(batch_size, in_features)
    >>> outputs, kl_divs = model(
    >>>     input,
    >>>     num_mc_samples=num_mc_samples,
    >>>     exact_kl_div=False,
    >>> )

    """

    # override
    def __init__(
        self,
        model: "nn.Module",
        prior_builder: "Callable[..., Distribution]",
        prior_kwargs: "Dict[str, Any]",
        posterior_builder: "Callable[..., Distribution]",
        posterior_kwargs: "Dict[str, Any]",
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
        posterior_builder:
            The posterior builder, i.e. a callable that receives
            keyword arguments and returns a posterior.
        posterior_kwargs:
            The keyword arguments to pass to the posterior builder.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__(model, prior_builder, prior_kwargs)
        self.posterior_builder = posterior_builder
        self.posterior_kwargs = posterior_kwargs

        # Extract particle
        # particle = model parameters flattened into a 1D vector
        particle = nn.utils.parameters_to_vector(model.parameters())

        # Build posterior (WITH gradient propagation)
        for k, v in posterior_kwargs.items():
            parameter = None
            if isinstance(v, (int, float)):
                parameter = Parameter(torch.full_like(particle, v))
            elif isinstance(v, Tensor):
                parameter = v.to(particle.device)
            if parameter is not None:
                self.register_parameter(f"posterior_{k}", parameter)
                posterior_kwargs[k] = parameter
        self._posterior = posterior_builder(**posterior_kwargs)

        # Adjust posterior event shape
        batch_shape = self._posterior.batch_shape
        if len(batch_shape) > 0:
            self._posterior = Independent(self._posterior, len(batch_shape))
        event_shape = self._posterior.event_shape
        if event_shape != particle.shape:
            raise ValueError(
                f"Posterior event size ({event_shape.numel()}) must be equal to "
                f"the number of model parameters ({particle.shape.numel()}) "
            )

    # override
    def named_parameters(
        self, prefix: "str" = "", recurse: "bool" = True
    ) -> "Iterator[Tuple[str, Parameter]]":
        return super().named_parameters(recurse=False)

    # override
    def state_dict(
        self, *args: "Any", **kwargs: "Any"
    ) -> "Dict[str, Optional[Tensor]]":
        tmp = super().state_dict(keep_vars=True)
        # Remove underlying model parameters but keep buffers
        for key in list(tmp.keys()):
            if ("prior" not in key and "posterior" not in key) and tmp[
                key
            ].requires_grad:
                tmp.pop(key)
        result = super().state_dict(*args, **kwargs)
        result = {k: v for k, v in result.items() if k in tmp}
        return result

    # override
    def load_state_dict(
        self, *args: "Any", strict: "bool" = True, **kwargs: "Any"
    ) -> "Tuple[List[str], List[str]]":
        return super().load_state_dict(*args, strict=False, **kwargs)

    # override
    def forward(
        self,
        *args: "Any",
        num_mc_samples: "int" = 1,
        exact_kl_div: "bool" = False,
        **kwargs: "Any",
    ) -> "Tuple[Any, Tensor]":
        """Forward pass.

        In the following, let `N` denote the number of Monte Carlo samples,
        `B = {B_1, ..., B_k}` the batch shape, and `O = {O_1, ..., O_m}`
        the shape of a leaf value of the underlying model output (can be
        a nested tensor).

        Parameters
        ----------
        args:
            The positional arguments to pass to the underlying model.
        num_mc_samples:
            The number of Monte Carlo samples.
        exact_kl_div:
            True to use the exact Kullback-Leibler divergence between
            parameter posterior and parameter prior (if a closed-form
            expression exists), False to use Monte Carlo approximation.
            For consistency, the scalar returned by the closed-form
            Kullback-Leibler divergence is expanded to a tensor of
            shape: ``[N]``.
        kwargs:
            The keyword arguments to pass to the underlying model.

        Returns
        -------
            - The outputs, shape of a leaf value: ``[N, *B, *O]``;
            - the Kullback-Leibler divergences between parameter
              posterior and parameter prior, shape: ``[N]``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if num_mc_samples < 1 or not float(num_mc_samples).is_integer():
            raise ValueError(
                f"`num_mc_samples` ({num_mc_samples}) must be in the integer interval [1, inf)"
            )

        # Replicate model (one replica for each Monte Carlo sample)
        models = nn.ModuleList(
            [self.model]
            + [copy.deepcopy(self.model) for _ in range(num_mc_samples - 1)]
        )

        # Sample new particles
        new_particles = self._posterior.rsample((num_mc_samples,))

        # Compute Kullback-Leibler divergences
        kl_divs = None
        if exact_kl_div:
            try:
                kl_divs = kl_divergence(self._posterior, self._prior).expand(
                    num_mc_samples
                )
            except NotImplementedError:
                kl_divs = None
                _LOGGER.warning(
                    "Could not compute exact Kullback-Leibler divergence, "
                    "reverting to Monte Carlo approximation"
                )
        if kl_divs is None:
            log_posteriors = self._posterior.log_prob(new_particles)
            log_priors = self._prior.log_prob(new_particles)
            kl_divs = log_posteriors - log_priors

        # Inject sampled particles
        new_particles = new_particles.flatten()
        start_idx = 0
        for parameter in models.parameters():
            num_elements = parameter.numel()
            new_parameter = new_particles[
                start_idx : start_idx + num_elements
            ].reshape_as(parameter)
            parameter.detach_().requires_grad_(False).copy_(new_parameter)
            start_idx += num_elements

        # Forward pass
        outputs = [model(*args, **kwargs) for model in models]

        return nested_stack(outputs), kl_divs

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(model: {self.model}, "
            f"prior: {self._prior}, "
            f"posterior: {self._posterior})"
        )
