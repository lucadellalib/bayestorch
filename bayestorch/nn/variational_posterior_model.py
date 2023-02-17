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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch import Tensor
from torch.distributions import Distribution, kl_divergence
from torch.nn import Module, Parameter

from bayestorch.nn.prior_model import PriorModel
from bayestorch.nn.utils import nested_apply


__all__ = [
    "VariationalPosteriorModel",
]


_T = TypeVar("_T", bound="VariationalPosteriorModel")

_T_destination = Module.T_destination

_IncompatibleKeys = torch.nn.modules.module._IncompatibleKeys

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
    >>> from bayestorch.nn import VariationalPosteriorModel
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
    >>> output = model(
    ...     input,
    ...     num_mc_samples=num_mc_samples,
    ... )
    >>> outputs, kl_divs = model(
    ...     input,
    ...     num_mc_samples=num_mc_samples,
    ...     return_kl_div=True,
    ...     reduction="none",
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
        posterior_builder:
            The posterior builder, i.e. a callable that receives
            keyword arguments and returns a posterior.
        posterior_kwargs:
            The keyword arguments to pass to the posterior builder.
            Tensor arguments are internally registered as parameters
            if their `requires_grad` attribute is True, as persistent
            buffers otherwise.
        model_parameters:
            The model parameters over which the prior and posterior
            are defined. Useful to selectively define a prior and a
            posterior over a restricted subset of modules/parameters.
            Default to ``model.parameters()``.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__(model, prior_builder, prior_kwargs, model_parameters)
        self.posterior_builder = posterior_builder
        self.posterior_kwargs = posterior_kwargs = {
            k: v for k, v in posterior_kwargs.items()
        }  # Avoid side effects

        # Build prior
        self.posterior = self._build_distribution(
            "posterior", posterior_builder, posterior_kwargs
        )

        # Retrieve indices of the selected parameters
        self._model_parameter_idxes = []
        all_model_parameters = list(model.parameters())
        for parameter in self.model_parameters:
            for i, x in enumerate(all_model_parameters):
                if parameter is x:
                    self._model_parameter_idxes.append(i)
                    break

        # Log Kullback-Leibler divergence warning only once
        self._log_kl_div_warning = True

    # override
    def named_parameters(
        self, *args: "Any", **kwargs: "Any"
    ) -> "Iterator[Tuple[str, Parameter]]":
        return (
            (k, v)
            for k, v in super().named_parameters(*args, **kwargs)
            if not any(v is parameter for parameter in self.model_parameters)
        )

    # override
    def state_dict(
        self,
        *args,
        destination: "_T_destination" = None,
        prefix: "str" = "",
        keep_vars: "bool" = False,
    ) -> "_T_destination":
        result = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=True
        )
        for k, v in list(result.items()):
            if any(v is parameter for parameter in self.model_parameters):
                result.pop(k)
            elif not keep_vars:
                result[k] = v.detach()
        return result

    # override
    def load_state_dict(
        self,
        state_dict: "Dict[str, Any]",
        strict: "bool" = True,
    ) -> "_IncompatibleKeys":
        parameter_names = [f"model.{name}" for name, _ in self.model.named_parameters()]
        for idx, parameter in zip(self._model_parameter_idxes, self.model_parameters):
            state_dict[parameter_names[idx]] = parameter
        result = super().load_state_dict(state_dict, strict)
        return result

    # override
    def to(self, *args: "Any", **kwargs: "Any") -> "_T":
        super().to(*args, **kwargs)

        # Rebuild posterior with parameters on new device
        self.posterior = self._build_distribution(
            "posterior", self.posterior_builder, self.posterior_kwargs
        )

        return self

    # override
    def forward(
        self,
        *args: "Any",
        num_mc_samples: "int" = 1,
        return_kl_div: "bool" = False,
        exact_kl_div: "bool" = False,
        reduction: "str" = "mean",
        **kwargs: "Any",
    ) -> "Union[Any, Tuple[Any, Tensor]]":
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
        return_kl_div:
            True to additionally return the Kullback-Leibler divergence of
            the prior from the posterior (usually required during training),
            False otherwise.
        exact_kl_div:
            True to use the exact Kullback-Leibler divergence of the prior
            from the posterior (if a closed-form expression exists),
            False to use Monte Carlo approximation.
        reduction:
            The reduction to apply to the leaf values of the underlying
            model output and to the Kullback-Leibler divergence (if
            `return_kl_div` is True) across Monte Carlo samples.
            Must be one of the following:
            - "none": no reduction is applied;
            - "mean": the leaf values and the Kullback-Leibler divergence
                      are averaged across Monte Carlo samples.
        kwargs:
            The keyword arguments to pass to the underlying model.

        Returns
        -------
            - The output, shape of a leaf value: ``[N, *B, *O]``
              if `reduction` is "none" , ``[*B, *O]`` otherwise;
            - if `return_kl_div` is True, the Kullback-Leibler
              divergence of the prior from the posterior, shape:
              ``[N]`` if `reduction` is "none" , ``[]`` otherwise.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        Warnings
        --------
        High memory usage is to be expected as `num_mc_samples - 1`
        full copies of the model must be maintained internally.

        """
        if num_mc_samples < 1 or not float(num_mc_samples).is_integer():
            raise ValueError(
                f"`num_mc_samples` ({num_mc_samples}) must be in the integer interval [1, inf)"
            )
        if reduction not in ["none", "mean"]:
            raise ValueError(
                f"`reduction` ({reduction}) must be one of {['none', 'mean']}"
            )

        if not torch.is_grad_enabled():
            return self._fast_forward(
                *args,
                num_mc_samples=num_mc_samples,
                return_kl_div=return_kl_div,
                exact_kl_div=exact_kl_div,
                reduction=reduction,
                **kwargs,
            )

        # Sample new particles
        new_particles = self.posterior.rsample((num_mc_samples,))

        # Replicate model (one replica for each Monte Carlo sample)
        models = [self.model] + [
            copy.deepcopy(self.model) for _ in range(num_mc_samples - 1)
        ]

        # Inject sampled particles
        new_particles = new_particles.flatten()
        start_idx = 0
        for model in models:
            all_model_parameters = list(model.parameters())
            model_parameters = [
                all_model_parameters[idx] for idx in self._model_parameter_idxes
            ]
            for parameter in model_parameters:
                end_idx = start_idx + parameter.numel()
                new_parameter = new_particles[start_idx:end_idx].reshape_as(parameter)
                parameter.detach_().requires_grad_(False).copy_(new_parameter)
                start_idx = end_idx

        # Forward pass
        outputs = [model(*args, **kwargs) for model in models]
        if reduction == "none":
            outputs = nested_apply(torch.stack, outputs)
        elif reduction == "mean":
            outputs = nested_apply(
                lambda inputs, dim: torch.mean(torch.stack(inputs, dim), dim), outputs
            )

        if not return_kl_div:
            return outputs

        # Compute Kullback-Leibler divergence
        kl_divs = None
        if exact_kl_div:
            try:
                kl_divs = kl_divergence(self.posterior, self.prior)
                if reduction == "none":
                    kl_divs = kl_divs.expand(num_mc_samples)
            except NotImplementedError:
                kl_divs = None
                if self._log_kl_div_warning:
                    _LOGGER.warning(
                        "Could not compute exact Kullback-Leibler divergence, "
                        "reverting to Monte Carlo approximation"
                    )
                    self._log_kl_div_warning = False
        if kl_divs is None:
            new_particles = new_particles.reshape(-1, *self.posterior.event_shape)
            log_posteriors = self.posterior.log_prob(new_particles)
            log_priors = self.prior.log_prob(new_particles)
            kl_divs = log_posteriors - log_priors
            if reduction == "mean":
                kl_divs = kl_divs.mean()

        return outputs, kl_divs

    # This implementation does not require copying the model
    # and can be used when gradient tracking is disabled
    def _fast_forward(
        self,
        *args: "Any",
        num_mc_samples: "int" = 1,
        return_kl_div: "bool" = False,
        exact_kl_div: "bool" = False,
        reduction: "str" = "mean",
        **kwargs: "Any",
    ) -> "Union[Any, Tuple[Any, Tensor]]":
        kl_divs = []
        if return_kl_div and exact_kl_div:
            try:
                kl_divs = kl_divergence(self.posterior, self.prior)
                if reduction == "none":
                    kl_divs = kl_divs.expand(num_mc_samples)
            except NotImplementedError:
                kl_divs = []
                if self._log_kl_div_warning:
                    _LOGGER.warning(
                        "Could not compute exact Kullback-Leibler divergence, "
                        "reverting to Monte Carlo approximation"
                    )
                    self._log_kl_div_warning = False

        outputs = []
        for _ in range(num_mc_samples):
            # Sample new particle
            new_particle = self.posterior.rsample()

            # Inject sampled particle
            start_idx = 0
            for parameter in self.model_parameters:
                end_idx = start_idx + parameter.numel()
                new_parameter = new_particle[start_idx:end_idx].reshape_as(parameter)
                parameter.detach_().requires_grad_(False).copy_(new_parameter)
                start_idx = end_idx

            # Forward pass
            output = self.model(*args, **kwargs)
            outputs.append(output)

            if isinstance(kl_divs, Tensor) or not return_kl_div:
                continue

            # Compute Kullback-Leibler divergence
            log_posterior = self.posterior.log_prob(new_particle)
            log_prior = self.prior.log_prob(new_particle)
            kl_div = log_posterior - log_prior
            kl_divs.append(kl_div)

        if reduction == "none":
            outputs = nested_apply(torch.stack, outputs)
        elif reduction == "mean":
            outputs = nested_apply(
                lambda inputs, dim: torch.mean(torch.stack(inputs, dim), dim), outputs
            )

        if not return_kl_div:
            return outputs

        if isinstance(kl_divs, list):
            kl_divs = torch.stack(kl_divs)
            if reduction == "mean":
                kl_divs = kl_divs.mean()

        return outputs, kl_divs

    # override
    def __repr__(self) -> "str":
        return (
            f"{type(self).__name__}"
            f"(model: {self.model}, "
            f"prior: {self.prior}, "
            f"posterior: {self.posterior}, "
            f"model_parameters: {sum(parameter.numel() for parameter in self.model_parameters)})"
        )
