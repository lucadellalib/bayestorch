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

"""Stochastic gradient Langevin dynamics optimizer."""

from typing import Any, Dict, Iterable, Union

import torch
from torch import Tensor
from torch.optim import Optimizer


__all__ = [
    "SGLD",
]


# Adapted from:
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/1f867a5bcbd1abfecede99807eb0b5f97ed8be7c/src/Stochastic_Gradient_Langevin_Dynamics/optimizers.py#L1
class SGLD(Optimizer):
    """Stochastic gradient Langevin dynamics optimizer.

    The optimization parameters are viewed as a posterior
    sample under stochastic gradient Langevin dynamics with
    noise rescaled in each dimension according to RMSProp.

    References
    ----------
    .. [1] C. Li, C. Chen, D. Carlson, and L. Carin.
           "Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks".
           In: AAAI. 2016, pp. 1788-1794.
           URL: https://arxiv.org/abs/1512.07666

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>>
    >>> from bayestorch.optimizers import SGLD
    >>>
    >>>
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> optimizer = SGLD(model.parameters())
    >>> input = torch.rand(batch_size, in_features)
    >>> output = model(input)
    >>> loss = output.sum()
    >>> loss.backward()
    >>> optimizer.step()

    """

    # override
    def __init__(
        self,
        params: "Union[Iterable[Tensor], Iterable[Dict[str, Any]]]",
        lr: "float" = 1e-2,
        num_burn_in_steps: "int" = 3000,
        precondition_decay_rate: "float" = 0.95,
        epsilon: "float" = 1e-8,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        params:
            The parameters to optimize.
        lr:
            The learning rate.
        num_burn_in_steps:
            The number of steps for which gradient
            statistics are collected to update the
            preconditioner before starting to draw
            noisy samples.
        precondition_decay_rate:
            The exponential decay rate for rescaling the
            preconditioner according to RMSProp. Should
            be close to 1 to approximate sampling from
            the posterior.
        epsilon:
            The term for improving numerical stability.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if lr < 0.0:
            raise ValueError(f"`lr` ({lr}) must be in the interval [0, inf)")
        if num_burn_in_steps < 0 or not float(num_burn_in_steps).is_integer():
            raise ValueError(
                f"`num_burn_in_steps` ({num_burn_in_steps}) must be in the integer interval [0, inf)"
            )
        if precondition_decay_rate < 0.0 or precondition_decay_rate > 1.0:
            raise ValueError(
                f"`precondition_decay_rate` ({precondition_decay_rate}) must be in the interval [0, 1]"
            )
        if epsilon <= 0.0:
            raise ValueError(f"`epsilon` ({epsilon}) must be in the interval (0, inf)")

        defaults = {
            "lr": lr,
            "num_burn_in_steps": int(num_burn_in_steps),
            "precondition_decay_rate": precondition_decay_rate,
            "epsilon": epsilon,
        }
        super().__init__(params, defaults)

    # override
    @torch.no_grad()
    def step(self) -> "None":
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]

                # State initialization
                if not state:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(param)

                lr = group["lr"]
                num_burn_in_steps = group["num_burn_in_steps"]
                precondition_decay_rate = group["precondition_decay_rate"]
                epsilon = group["epsilon"]

                iteration = state["iteration"]
                momentum = state["momentum"]

                iteration += 1
                grad = param.grad

                # Momentum update
                momentum += (1.0 - precondition_decay_rate) * (grad**2 - momentum)

                # Burn-in steps
                if iteration <= num_burn_in_steps:
                    stddev = torch.zeros_like(param)
                else:
                    stddev = 1.0 / torch.full_like(param, lr).sqrt()

                # Draw random sample
                preconditioner = 1.0 / (momentum + epsilon).sqrt()
                mean = 0.5 * preconditioner * grad
                stddev *= preconditioner.sqrt()
                sample = torch.normal(mean, stddev)

                # Parameter update
                param += -lr * sample
