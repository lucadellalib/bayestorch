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

"""Inverse softplus scale normal distribution."""

from typing import Optional, Union

import torch.nn.functional as F
from torch import Size, Tensor
from torch.distributions import Normal, constraints


__all__ = [
    "SoftplusInvScaleNormal",
]


class SoftplusInvScaleNormal(Normal):
    """Normal distribution parameterized by location
    and inverse softplus scale parameters.

    Scale parameter is computed as `softplus(softplus_inv_scale)`.

    Examples
    --------
    >>> from bayestorch.distributions import SoftplusInvScaleNormal
    >>>
    >>>
    >>> loc = 0.0
    >>> softplus_inv_scale = -1.0
    >>> distribution = SoftplusInvScaleNormal(loc, softplus_inv_scale)

    """

    arg_constraints = {
        "loc": constraints.real,
        "softplus_inv_scale": constraints.real,
    }  # override

    # override
    def __init__(
        self,
        loc: "Union[int, float, Tensor]",
        softplus_inv_scale: "Union[int, float, Tensor]",
        validate_args: "Optional[bool]" = None,
    ) -> "None":
        super().__init__(loc, softplus_inv_scale, validate_args)

    # override
    @property
    def scale(self) -> "Tensor":
        return F.softplus(self.softplus_inv_scale)

    # override
    @scale.setter
    def scale(self, value: "Tensor") -> "None":
        self.softplus_inv_scale = value

    # override
    def expand(
        self,
        batch_shape: "Size" = Size(),  # noqa: B008
        _instance: "Optional[SoftplusInvScaleNormal]" = None,
    ) -> "SoftplusInvScaleNormal":
        new = self._get_checked_instance(SoftplusInvScaleNormal, _instance)
        loc = self.loc.expand(batch_shape)
        softplus_inv_scale = self.softplus_inv_scale.expand(batch_shape)
        super(SoftplusInvScaleNormal, new).__init__(
            loc, softplus_inv_scale, validate_args=False
        )
        new._validate_args = self._validate_args
        return new
