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

"""Model utilities."""

from typing import Dict, Sequence, TypeVar, Union

import torch
from torch import Tensor


__all__ = [
    "nested_stack",
]


_T = TypeVar("_T")

_Nested = Union[_T, Sequence[_T], Dict[str, _T]]


def nested_stack(
    inputs: "Sequence[_Nested[Tensor]]", dim: "int" = 0
) -> "_Nested[Tensor]":
    """Stack a sequence of possibly nested tensors
    along a dimension.

    Parameters
    ----------
    inputs:
        The sequence of possibly nested tensors.
    dim:
        The dimension.

    Examples
    --------
    >>> import torch
    >>>
    >>> from bayestorch.models.utils import nested_stack
    >>>
    >>>
    >>> num_outputs = 4
    >>> inputs = [
    ...    {"a": [torch.rand(2, 3), torch.rand(3, 5)], "b": torch.rand(1, 2)}
    ...    for _ in range(num_outputs)
    ... ]
    >>> outputs = nested_stack(inputs)

    """
    first_input = inputs[0]
    if isinstance(first_input, Tensor):
        return torch.stack(inputs, dim=dim)
    if first_input is None:
        return None
    if isinstance(first_input, dict):
        return type(first_input)(
            (k, nested_stack([output[k] for output in inputs])) for k in first_input
        )
    return type(first_input)(map(nested_stack, zip(*inputs)))
