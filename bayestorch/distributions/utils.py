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

"""Distribution utilities."""

from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from torch import Tensor, device, nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    Laplace,
    MixtureSameFamily,
    Normal,
)

from bayestorch.distributions.deterministic import Deterministic
from bayestorch.distributions.log_scale_normal import LogScaleNormal
from bayestorch.distributions.softplus_inv_scale_normal import SoftplusInvScaleNormal


__all__ = [
    "get_deterministic",
    "get_laplace",
    "get_loc_scale",
    "get_log_scale_normal",
    "get_mixture_laplace",
    "get_mixture_loc_scale",
    "get_mixture_log_scale_normal",
    "get_mixture_normal",
    "get_mixture_softplus_inv_scale_normal",
    "get_normal",
    "get_softplus_inv_scale_normal",
]


_T = TypeVar("_T", bound=Distribution)


def get_loc_scale(
    distribution_cls: "Type[_T]",
    parameters: "Iterable[Tensor]",
    loc: "float" = 0.0,
    scale: "float" = 1.0,
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor], _T], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    location-scale distribution over the parameters of a model.

    Parameters
    ----------
    distribution_cls:
        The location-scale distribution class.
    parameters:
        The parameters.
    loc:
        The distribution location.
    scale:
        The distribution scale.
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters, False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The location-scale distribution builder;
        - the location-scale distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_loc_scale
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_loc_scale(
    ...     distributions.Normal, model.parameters(), 0.0, 1.0, requires_grad=True,
    ... )
    >>> distribution = builder(**kwargs)

    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError(f"`parameters` ({parameters}) must be non-empty")
    num_parameters = sum(parameter.numel() for parameter in parameters)
    loc = torch.full((num_parameters,), loc, device=device, requires_grad=requires_grad)
    scale = torch.full(
        (num_parameters,),
        scale,
        device=device,
        requires_grad=requires_grad,
    )
    return lambda **kwargs: distribution_cls(
        loc=kwargs[f"{prefix}loc"], scale=kwargs[f"{prefix}scale"]
    ), {
        f"{prefix}loc": loc,
        f"{prefix}scale": scale,
    }


def get_mixture_loc_scale(
    distribution_cls: "Type[_T]",
    parameters: "Iterable[Tensor]",
    weights: "Sequence[float]" = (0.75, 0.25),
    locs: "Sequence[float]" = (0.0, 0.0),
    scales: "Sequence[float]" = (1.0, 2.0),
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor, Tensor], MixtureSameFamily], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    location-scale mixture distribution over the parameters of a model.

    Parameters
    ----------
    distribution_cls:
        The location-scale distribution class.
    parameters:
        The parameters.
    weights:
        The mixture weights
        (one for each mixture component).
    locs:
        The distribution locations
        (one for each mixture component).
    scales:
        The distribution scales
        (one for each mixture component).
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters (`weights` excluded),
        False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The location-scale mixture distribution builder;
        - the location-scale mixture distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_mixture_loc_scale
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_mixture_loc_scale(
    ...     distributions.Normal, model.parameters(), (0.75, 0.25), (0.0, 0.0), (1.0, 2.0), requires_grad=True,
    ... )
    >>> distribution = builder(**kwargs)

    """
    if sum(weights) != 1.0:
        raise ValueError(f"`weights` ({weights}) must sum to 1")
    if len(weights) != len(locs) or len(weights) != len(scales):
        raise ValueError(
            f"`weights` ({weights}), `locs` ({locs}) and `scales` ({scales}) must have the same length"
        )
    parameters = list(parameters)
    if not parameters:
        raise ValueError(f"`parameters` ({parameters}) must be non-empty")
    num_parameters = sum(parameter.numel() for parameter in parameters)
    weights = torch.as_tensor(weights, device=device)
    locs = torch.stack(
        [torch.full((num_parameters,), loc, device=device) for loc in locs]
    ).requires_grad_(requires_grad)
    scales = torch.stack(
        [torch.full((num_parameters,), scale, device=device) for scale in scales]
    ).requires_grad_(requires_grad)
    return (
        lambda **kwargs: MixtureSameFamily(
            Categorical(probs=kwargs[f"{prefix}weights"]),
            Independent(
                distribution_cls(
                    loc=kwargs[f"{prefix}locs"], scale=kwargs[f"{prefix}scales"]
                ),
                1,
            ),
        ),
        {f"{prefix}weights": weights, f"{prefix}locs": locs, f"{prefix}scales": scales},
    )


def get_deterministic(
    parameters: "Iterable[Tensor]",
    value: "Optional[float]" = None,
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor], Deterministic], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    deterministic distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    value:
        The distribution value.
        Default to the current parameter values.
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters, False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The deterministic distribution builder;
        - the deterministic distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_deterministic
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_deterministic(model.parameters())
    >>> distribution = builder(**kwargs)

    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError(f"`parameters` ({parameters}) must be non-empty")
    if value is None:
        with torch.no_grad():
            value = nn.utils.parameters_to_vector(parameters).to(device)
        value.requires_grad_(requires_grad)
    else:
        num_parameters = sum(parameter.numel() for parameter in parameters)
        value = torch.full(
            (num_parameters,), value, device=device, requires_grad=requires_grad
        )
    return lambda **kwargs: Deterministic(value=kwargs[f"{prefix}value"]), {
        f"{prefix}value": value
    }


def get_normal(
    parameters: "Iterable[Tensor]",
    loc: "float" = 0.0,
    scale: "float" = 1.0,
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor], Normal], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    normal distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    loc:
        The distribution location.
    scale:
        The distribution scale.
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters, False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The normal distribution builder;
        - the normal distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_normal
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_normal(model.parameters(), 0.0, 1.0, requires_grad=True)
    >>> distribution = builder(**kwargs)

    """
    return get_loc_scale(Normal, parameters, loc, scale, device, requires_grad, prefix)


def get_log_scale_normal(
    parameters: "Iterable[Tensor]",
    loc: "float" = 0.0,
    log_scale: "float" = -1.0,
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor], LogScaleNormal], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    log scale normal distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    loc:
        The distribution location.
    log_scale:
        The distribution log scale.
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters, False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The log scale normal distribution builder;
        - the log scale normal distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_log_scale_normal
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_log_scale_normal(model.parameters(), 0.0, -1.0, requires_grad=True)
    >>> distribution = builder(**kwargs)

    """
    _, kwargs = get_loc_scale(
        LogScaleNormal,
        parameters,
        loc,
        log_scale,
        device,
        requires_grad,
        prefix,
    )
    kwargs[f"{prefix}log_scale"] = kwargs.pop(f"{prefix}scale")
    return (
        lambda **kwargs: LogScaleNormal(
            loc=kwargs[f"{prefix}loc"], log_scale=kwargs[f"{prefix}log_scale"]
        ),
        kwargs,
    )


def get_softplus_inv_scale_normal(
    parameters: "Iterable[Tensor]",
    loc: "float" = 0.0,
    softplus_inv_scale: "float" = -1.0,
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor], SoftplusInvScaleNormal], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of an inverse
    softplus scale normal distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    loc:
        The distribution location.
    softplus_inv_scale:
        The distribution inverse softplus scale.
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters, False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The inverse softplus scale normal distribution builder;
        - the inverse softplus scale normal distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_softplus_inv_scale_normal
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_softplus_inv_scale_normal(model.parameters(), 0.0, -1.0, requires_grad=True)
    >>> distribution = builder(**kwargs)

    """
    _, kwargs = get_loc_scale(
        SoftplusInvScaleNormal,
        parameters,
        loc,
        softplus_inv_scale,
        device,
        requires_grad,
        prefix,
    )
    kwargs[f"{prefix}softplus_inv_scale"] = kwargs.pop(f"{prefix}scale")
    return (
        lambda **kwargs: SoftplusInvScaleNormal(
            loc=kwargs[f"{prefix}loc"],
            softplus_inv_scale=kwargs[f"{prefix}softplus_inv_scale"],
        ),
        kwargs,
    )


def get_laplace(
    parameters: "Iterable[Tensor]",
    loc: "float" = 0.0,
    scale: "float" = 1.0,
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor], Laplace], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    Laplace distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    loc:
        The distribution location.
    scale:
        The distribution scale.
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters, False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The Laplace distribution builder;
        - the Laplace distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_laplace
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_laplace(model.parameters(), 0.0, 1.0, requires_grad=True)
    >>> distribution = builder(**kwargs)

    """
    return get_loc_scale(Laplace, parameters, loc, scale, device, requires_grad, prefix)


def get_mixture_normal(
    parameters: "Iterable[Tensor]",
    weights: "Sequence[float]" = (0.75, 0.25),
    locs: "Sequence[float]" = (0.0, 0.0),
    scales: "Sequence[float]" = (1.0, 2.0),
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor, Tensor], MixtureSameFamily], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    normal mixture distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    weights:
        The mixture weights
        (one for each mixture component).
    locs:
        The distribution locations
        (one for each mixture component).
    scales:
        The distribution scales
        (one for each mixture component).
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters (`weights` excluded),
        False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The normal mixture distribution builder;
        - the normal mixture distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_mixture_normal
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_mixture_normal(
    ...     model.parameters(), (0.75, 0.25), (0.0, 0.0), (1.0, 2.0), requires_grad=True,
    ... )
    >>> distribution = builder(**kwargs)

    """
    return get_mixture_loc_scale(
        Normal,
        parameters,
        weights,
        locs,
        scales,
        device,
        requires_grad,
        prefix,
    )


def get_mixture_log_scale_normal(
    parameters: "Iterable[Tensor]",
    weights: "Sequence[float]" = (0.75, 0.25),
    locs: "Sequence[float]" = (0.0, 0.0),
    log_scales: "Sequence[float]" = (-1.0, -2.0),
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor, Tensor], MixtureSameFamily], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a log
    scale normal mixture distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    weights:
        The mixture weights
        (one for each mixture component).
    locs:
        The distribution locations
        (one for each mixture component).
    log_scales:
        The distribution log scales
        (one for each mixture component).
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters (`weights` excluded),
        False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The log scale normal mixture distribution builder;
        - the log scale normal mixture distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_mixture_log_scale_normal
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_mixture_log_scale_normal(
    ...     model.parameters(), (0.75, 0.25), (0.0, 0.0), (-1.0, -2.0), requires_grad=True,
    ... )
    >>> distribution = builder(**kwargs)

    """
    _, kwargs = get_mixture_loc_scale(
        LogScaleNormal,
        parameters,
        weights,
        locs,
        log_scales,
        device,
        requires_grad,
        prefix,
    )
    kwargs[f"{prefix}log_scales"] = kwargs.pop(f"{prefix}scales")
    return (
        lambda **kwargs: MixtureSameFamily(
            Categorical(probs=kwargs[f"{prefix}weights"]),
            Independent(
                LogScaleNormal(
                    loc=kwargs[f"{prefix}locs"], log_scale=kwargs[f"{prefix}log_scales"]
                ),
                1,
            ),
        ),
        kwargs,
    )


def get_mixture_softplus_inv_scale_normal(
    parameters: "Iterable[Tensor]",
    weights: "Sequence[float]" = (0.75, 0.25),
    locs: "Sequence[float]" = (0.0, 0.0),
    softplus_inv_scales: "Sequence[float]" = (-1.0, -2.0),
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor, Tensor], MixtureSameFamily], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of an inverse
    softplus scale normal mixture distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    weights:
        The mixture weights
        (one for each mixture component).
    locs:
        The distribution locations
        (one for each mixture component).
    softplus_inv_scales:
        The distribution inverse softplus scales
        (one for each mixture component).
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters (`weights` excluded),
        False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The inverse softplus scale normal mixture distribution builder;
        - the inverse softplus scale normal mixture distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_mixture_softplus_inv_scale_normal
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_mixture_softplus_inv_scale_normal(
    ...     model.parameters(), (0.75, 0.25), (0.0, 0.0), (-1.0, -2.0), requires_grad=True,
    ... )
    >>> distribution = builder(**kwargs)

    """
    _, kwargs = get_mixture_loc_scale(
        SoftplusInvScaleNormal,
        parameters,
        weights,
        locs,
        softplus_inv_scales,
        device,
        requires_grad,
        prefix,
    )
    kwargs[f"{prefix}softplus_inv_scales"] = kwargs.pop(f"{prefix}scales")
    return (
        lambda **kwargs: MixtureSameFamily(
            Categorical(probs=kwargs[f"{prefix}weights"]),
            Independent(
                SoftplusInvScaleNormal(
                    loc=kwargs[f"{prefix}locs"],
                    softplus_inv_scale=kwargs[f"{prefix}softplus_inv_scales"],
                ),
                1,
            ),
        ),
        kwargs,
    )


def get_mixture_laplace(
    parameters: "Iterable[Tensor]",
    weights: "Sequence[float]" = (0.75, 0.25),
    locs: "Sequence[float]" = (0.0, 0.0),
    scales: "Sequence[float]" = (1.0, 2.0),
    device: "Optional[Union[device, str]]" = "cpu",
    requires_grad: "bool" = False,
    prefix: "str" = "",
) -> "Tuple[Callable[[Tensor, Tensor, Tensor], MixtureSameFamily], Dict[str, Tensor]]":
    """Return a builder and the corresponding keyword arguments of a
    Laplace mixture distribution over the parameters of a model.

    Parameters
    ----------
    parameters:
        The parameters.
    weights:
        The mixture weights
        (one for each mixture component).
    locs:
        The distribution locations
        (one for each mixture component).
    scales:
        The distribution scales
        (one for each mixture component).
    device:
        The device.
    requires_grad:
        True to enable gradient tracking on the
        distribution parameters (`weights` excluded),
        False otherwise.
    prefix:
        The prefix to prepend to each keyword argument.
        Useful to build a concatenated distribution.

    Returns
    -------
        - The Laplace mixture distribution builder;
        - the Laplace mixture distribution keyword arguments.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> from torch import distributions, nn
    >>>
    >>> from bayestorch.distributions.utils import get_mixture_laplace
    >>>
    >>>
    >>> model = nn.Linear(4, 2)
    >>> builder, kwargs = get_mixture_laplace(
    ...     model.parameters(), (0.75, 0.25), (0.0, 0.0), (1.0, 2.0), requires_grad=True,
    ... )
    >>> distribution = builder(**kwargs)

    """
    return get_mixture_loc_scale(
        Laplace,
        parameters,
        weights,
        locs,
        scales,
        device,
        requires_grad,
        prefix,
    )
