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

"""Stochastic gradient Hamiltonian Monte Carlo optimizer."""

from typing import Any, Dict, Iterable, Union

import torch
from torch import Tensor
from torch.optim import Optimizer


__all__ = [
    "SGHMC",
]


# Adapted from:
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/1f867a5bcbd1abfecede99807eb0b5f97ed8be7c/src/Stochastic_Gradient_HMC_SA/optimizers.py#L1
class SGHMC(Optimizer):
    """Stochastic gradient Hamiltonian Monte Carlo optimizer.

    A burn-in procedure is used to adapt the hyperparameters
    during the initial stages of sampling.

    References
    ----------
    .. [1] T. Chen, E. B. Fox, and C. Guestrin.
           "Stochastic Gradient Hamiltonian Monte Carlo".
           In: ICML. 2014, pp. 1683-1691.
           URL: https://arxiv.org/abs/1402.4102

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>>
    >>> from bayestorch.optimizers import SGHMC
    >>>
    >>>
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> model = nn.Linear(in_features, out_features)
    >>> optimizer = SGHMC(model.parameters())
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
        momentum_decay: "float" = 0.05,
        grad_noise: "float" = 0.0,
        epsilon: "float" = 1e-16,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        params:
            The parameters to optimize.
        lr:
            The learning rate.
        num_burn_in_steps:
            The number of burn-in steps. At each step,
            the optimizer hyperparameters are adapted
            to decrease the error.
        momentum_decay:
            The momentum decay per timestep.
        grad_noise:
            The constant per-parameter gradient
            noise used for sampling.
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
        if momentum_decay < 0.0:
            raise ValueError(
                f"`momentum_decay` ({momentum_decay}) must be in the interval [0, inf)"
            )
        if grad_noise < 0.0:
            raise ValueError(
                f"`grad_noise` ({grad_noise}) must be in the interval [0, inf)"
            )
        if epsilon <= 0.0:
            raise ValueError(f"`epsilon` ({epsilon}) must be in the interval (0, inf)")

        defaults = {
            "lr": lr,
            "num_burn_in_steps": int(num_burn_in_steps),
            "momentum_decay": momentum_decay,
            "grad_noise": grad_noise,
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
                    state["tau"] = torch.ones_like(param)
                    state["g"] = torch.ones_like(param)
                    state["v_hat"] = torch.ones_like(param)
                    state["momentum"] = torch.zeros_like(param)

                lr = group["lr"]
                num_burn_in_steps = group["num_burn_in_steps"]
                momentum_decay = group["momentum_decay"]
                grad_noise = group["grad_noise"]
                epsilon = group["epsilon"]

                iteration = state["iteration"]
                tau = state["tau"]
                g = state["g"]
                v_hat = state["v_hat"]
                momentum = state["momentum"]

                iteration += 1
                grad = param.grad
                r = 1.0 / (tau + 1.0)
                m_inv = 1.0 / v_hat.sqrt()

                # Burn-in steps
                if iteration <= num_burn_in_steps:
                    tau += 1.0 - tau * (g**2 / v_hat)
                    g += (grad - g) * r
                    v_hat += (grad**2 - v_hat) * r

                # Draw random sample
                grad_noise_var = (
                    2.0 * (lr**2) * momentum_decay * m_inv
                    - 2.0 * (lr**3) * (m_inv**2) * grad_noise
                    - (lr**4)
                )
                stddev = grad_noise_var.clamp(min=epsilon).sqrt()
                sample = torch.normal(0.0, stddev)

                # Parameter update
                momentum += sample - lr**2 * m_inv * grad - momentum_decay * momentum
                param += momentum
