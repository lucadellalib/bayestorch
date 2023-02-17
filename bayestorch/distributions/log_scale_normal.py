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

"""Log scale normal distribution."""

from typing import Optional, Union

from torch import Size, Tensor
from torch.distributions import Normal, constraints


__all__ = [
    "LogScaleNormal",
]


class LogScaleNormal(Normal):
    """Normal distribution parameterized by location
    and log scale parameters.

    Scale parameter is computed as `exp(log_scale)`.

    Examples
    --------
    >>> from bayestorch.distributions import LogScaleNormal
    >>>
    >>>
    >>> loc = 0.0
    >>> log_scale = -1.0
    >>> distribution = LogScaleNormal(loc, log_scale)

    """

    arg_constraints = {
        "loc": constraints.real,
        "log_scale": constraints.real,
    }  # override

    # override
    def __init__(
        self,
        loc: "Union[int, float, Tensor]",
        log_scale: "Union[int, float, Tensor]",
        validate_args: "Optional[bool]" = None,
    ) -> "None":
        super().__init__(loc, log_scale, validate_args)

    @property
    def mode(self) -> "Tensor":
        return self.mean

    # override
    @property
    def scale(self) -> "Tensor":
        return self.log_scale.exp()

    # override
    @scale.setter
    def scale(self, value: "Tensor") -> "None":
        self.log_scale = value

    # override
    def expand(
        self,
        batch_shape: "Size" = Size(),  # noqa: B008
        _instance: "Optional[LogScaleNormal]" = None,
    ) -> "LogScaleNormal":
        new = self._get_checked_instance(LogScaleNormal, _instance)
        loc = self.loc.expand(batch_shape)
        log_scale = self.log_scale.expand(batch_shape)
        super(LogScaleNormal, new).__init__(loc, log_scale, validate_args=False)
        new._validate_args = self._validate_args
        return new
