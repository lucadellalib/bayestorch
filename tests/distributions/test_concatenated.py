#!/usr/bin/env python3

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

"""Test concatenated distribution."""

import pytest
import torch
from torch.distributions import Categorical, Normal

from bayestorch.distributions import Concatenated


def test_concatenated() -> "None":
    loc = 0.0
    scale = 1.0
    logits = torch.as_tensor([0.25, 0.15, 0.10, 0.30, 0.20])
    distribution = Concatenated([Normal(loc, scale), Categorical(logits)])
    print(distribution)
    print(f"Mean: {distribution.mean}")
    print(f"Standard deviation: {distribution.stddev}")


if __name__ == "__main__":
    pytest.main([__file__])
