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

from torch import Tensor
from torch.distributions import Normal


__all__ = [
    "LogScaleNormal",
]


class LogScaleNormal(Normal):
    """Normal distribution parameterized by `loc`
    and `log_scale` parameters.

    Examples
    --------
    >>> loc = 0.0
    >>> log_scale = -1.0
    >>> distribution = LogScaleNormal(loc, log_scale)

    """

    # override
    def __init__(
        self,
        loc: "Union[int, float, Tensor]",
        log_scale: "Union[int, float, Tensor]",
        validate_args: "Optional[bool]" = None,
    ) -> "None":
        super().__init__(loc, log_scale, validate_args)

    # override
    @property
    def scale(self) -> "Tensor":
        return self._log_scale.exp()

    # override
    @scale.setter
    def scale(self, value: "Tensor") -> "None":
        self._log_scale = value
