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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

from torch import Tensor, nn
from torch.distributions import Distribution, kl_divergence
from torch.nn import Module, Parameter

from bayestorch.models.prior_model import PriorModel
from bayestorch.models.utils import nested_stack


__all__ = [
    "VariationalPosteriorModel",
]


_T = TypeVar("_T", bound="VariationalPosteriorModel")

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
    >>>
    >>> from bayestorch.distributions import LogScaleNormal, SoftplusInvScaleNormal
    >>> from bayestorch.models import VariationalPosteriorModel
    >>>
    >>>
    >>> num_mc_samples = 5
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> num_parameters = sum(parameter.numel() for parameter in model.parameters())
    >>> model = VariationalPosteriorModel(
    ...     model,
    ...     prior_builder=LogScaleNormal,
    ...     prior_kwargs={
    ...         "loc": torch.zeros(num_parameters),
    ...         "log_scale": torch.full((num_parameters,), -1.0),
    ...     },
    ...     posterior_builder=SoftplusInvScaleNormal,
    ...     posterior_kwargs={
    ...         "loc": torch.zeros(num_parameters, requires_grad=True),
    ...         "softplus_inv_scale": torch.full((num_parameters,), -7.0, requires_grad=True),
    ...     },
    ... )
    >>> input = torch.rand(batch_size, in_features)
    >>> outputs, kl_divs = model(
    ...     input,
    ...     num_mc_samples=num_mc_samples,
    ...     exact_kl_div=False,
    ... )

    """

    posterior: "Distribution"
    """The posterior distribution."""

    # override
    def __init__(
        self,
        model: "Module",
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
            Tensor arguments are internally registered as
            parameters if their `requires_grad` attribute
            is True, as persistent buffers otherwise.
        posterior_builder:
            The posterior builder, i.e. a callable that receives
            keyword arguments and returns a posterior.
        posterior_kwargs:
            The keyword arguments to pass to the posterior builder.
            Tensor arguments are internally registered as
            parameters if their `requires_grad` attribute
            is True, as persistent buffers otherwise.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__(model, prior_builder, prior_kwargs)
        self.posterior_builder = posterior_builder
        self.posterior_kwargs = posterior_kwargs = {
            k: v for k, v in posterior_kwargs.items()
        }  # Avoid side effects

        # Build prior
        self.posterior = self._build_distribution(
            "posterior", posterior_builder, posterior_kwargs
        )

        # Log Kullback-Leibler divergence warning only once
        self._log_kl_div_warning = True

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
    def to(self, *args: "Any", **kwargs: "Any") -> "_T":
        super().to(*args, **kwargs)

        # Rebuild posterior with parameters on new device
        self.prior = self._build_distribution(
            "prior", self.prior_builder, self.prior_kwargs
        )

        return self

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
            posterior and prior (if a closed-form expression exists),
            False to use Monte Carlo approximation.
            For consistency, the scalar returned by the closed-form
            Kullback-Leibler divergence is expanded to a tensor of
            shape: ``[N]``.
        kwargs:
            The keyword arguments to pass to the underlying model.

        Returns
        -------
            - The outputs, shape of a leaf value: ``[N, *B, *O]``;
            - the Kullback-Leibler divergences between posterior
              and prior, shape: ``[N]``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if num_mc_samples < 1 or not float(num_mc_samples).is_integer():
            raise ValueError(
                f"`num_mc_samples` ({num_mc_samples}) must be in the integer interval [1, inf)"
            )

        # Sample new particles
        new_particles = self.posterior.rsample((num_mc_samples,))

        # Compute Kullback-Leibler divergences
        kl_divs = None
        if exact_kl_div:
            try:
                kl_divs = kl_divergence(self.posterior, self.prior).expand(
                    num_mc_samples
                )
            except NotImplementedError:
                kl_divs = None
                if self._log_kl_div_warning:
                    _LOGGER.warning(
                        "Could not compute exact Kullback-Leibler divergence, "
                        "reverting to Monte Carlo approximation"
                    )
                    self._log_kl_div_warning = False
        if kl_divs is None:
            log_posteriors = self.posterior.log_prob(new_particles)
            log_priors = self.prior.log_prob(new_particles)
            kl_divs = log_posteriors - log_priors

        # Replicate model (one replica for each Monte Carlo sample)
        models = nn.ModuleList(
            [self.model]
            + [copy.deepcopy(self.model) for _ in range(num_mc_samples - 1)]
        )

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
            f"prior: {self.prior}, "
            f"posterior: {self.posterior})"
        )
