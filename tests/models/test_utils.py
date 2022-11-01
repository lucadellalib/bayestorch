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

"""Test model utilities."""

import pytest
import torch

from bayestorch.models.utils import nested_stack


def test_nested_stack() -> "None":
    num_outputs = 4
    inputs = [
        {"a": [torch.rand(2, 3), torch.rand(3, 5)], "b": torch.rand(1, 2)}
        for _ in range(num_outputs)
    ]
    outputs = nested_stack(inputs)
    print(f"Shape of first nested input: {inputs[0]['a'][0].shape}")
    print(f"Shape of first nested output: {outputs['a'][0].shape}")


if __name__ == "__main__":
    pytest.main([__file__])
