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

"""Test distribution utilities."""

import pytest
import torch

from bayestorch.distributions.utils import (
    get_deterministic,
    get_laplace,
    get_log_scale_normal,
    get_mixture_laplace,
    get_mixture_log_scale_normal,
    get_mixture_normal,
    get_mixture_softplus_inv_scale_normal,
    get_normal,
    get_softplus_inv_scale_normal,
)


def test_get_deterministic() -> "None":
    model = torch.nn.Linear(4, 2)
    _, _ = get_deterministic(model.parameters(), 2.0, requires_grad=True)
    builder, kwargs = get_deterministic(model.parameters(), requires_grad=True)
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_laplace() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_laplace(model.parameters(), 0.0, 1.0, requires_grad=True)
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_normal() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_normal(model.parameters(), 0.0, 1.0, requires_grad=True)
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_log_scale_normal() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_log_scale_normal(
        model.parameters(), 0.0, -1.0, requires_grad=True
    )
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_softplus_inv_scale_normal() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_softplus_inv_scale_normal(
        model.parameters(), 0.0, -1.0, requires_grad=True
    )
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_mixture_laplace() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_mixture_laplace(
        model.parameters(),
        (0.75, 0.25),
        (0.0, 0.0),
        (1.0, 2.0),
        requires_grad=True,
    )
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_mixture_normal() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_mixture_normal(
        model.parameters(),
        (0.75, 0.25),
        (0.0, 0.0),
        (1.0, 2.0),
        requires_grad=True,
    )
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_mixture_log_scale_normal() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_mixture_log_scale_normal(
        model.parameters(),
        (0.75, 0.25),
        (0.0, 0.0),
        (-1.0, -2.0),
        requires_grad=True,
    )
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


def test_get_mixture_softplus_inv_scale_normal() -> "None":
    model = torch.nn.Linear(4, 2)
    builder, kwargs = get_mixture_softplus_inv_scale_normal(
        model.parameters(),
        (0.75, 0.25),
        (0.0, 0.0),
        (-1.0, -2.0),
        requires_grad=True,
    )
    distribution = builder(**kwargs)
    print(
        f"Number of parameters: {sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(f"Sample shape: {distribution.sample().shape}")


if __name__ == "__main__":
    pytest.main([__file__])
