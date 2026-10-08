# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spherical-harmonics (l >= 1) evaluation whose backward applies the Adam step
to the SH coefficients directly, without materializing their gradient.

The shN gradient is only consumed by its optimizer, so fusing the update into
the SH backward saves writing and re-reading a [N, K, 3] gradient every step.
The Adam state lives in the regular ``torch.optim.Adam`` instance (``exp_avg``,
``exp_avg_sq``, ``step``), so densification utilities that edit optimizer state
keep working. ``shN.grad`` stays ``None``, which makes ``optimizer.step()`` skip
the parameter.
"""

import math

import torch
from torch import Tensor

from ..cuda._wrapper import _make_lazy_cuda_func, spherical_harmonics_l1_plus


class _SHL1PlusFusedAdam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, degree: int, means: Tensor, viewmats: Tensor, shN: Tensor, optimizer: torch.optim.Adam):
        ctx.degree = degree
        ctx.optimizer = optimizer
        ctx.save_for_backward(means, viewmats, shN)
        return spherical_harmonics_l1_plus(degree, means, viewmats, shN)

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        means, viewmats, shN = ctx.saved_tensors
        opt = ctx.optimizer
        (group,) = opt.param_groups
        (param,) = group["params"]
        assert param is shN, "optimizer must own exactly this shN parameter"
        state = opt.state[param]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["step"] += 1
        step = float(state["step"])
        beta1, beta2 = group["betas"]
        bias_correction1 = 1 - beta1**step
        bias_correction2_sqrt = math.sqrt(1 - beta2**step)
        v_means = _make_lazy_cuda_func("spherical_harmonics_l1_plus_bwd_adam")(
            ctx.degree,
            means.contiguous(),
            viewmats.contiguous(),
            shN.data,
            v_colors.contiguous(),
            state["exp_avg"],
            state["exp_avg_sq"],
            group["lr"] / bias_correction1,
            beta1,
            beta2,
            group["eps"],
            bias_correction2_sqrt,
            ctx.needs_input_grad[1],
        )
        return None, v_means, None, None, None


def spherical_harmonics_l1_plus_fused_adam(
    degree: int, means: Tensor, viewmats: Tensor, shN: Tensor, optimizer: torch.optim.Adam
) -> Tensor:
    """Like :func:`gsplat.spherical_harmonics_l1_plus` (dense, unbatched), but the
    backward pass performs ``optimizer``'s Adam step on ``shN`` in place.

    ``optimizer`` must be a ``torch.optim.Adam`` holding only ``shN`` with
    ``weight_decay=0`` and ``amsgrad=False``. Call this once per optimizer step.
    """
    (group,) = optimizer.param_groups
    assert group.get("weight_decay", 0) == 0 and not group.get("amsgrad", False)
    assert not group.get("maximize", False)
    return _SHL1PlusFusedAdam.apply(degree, means, viewmats, shN, optimizer)
