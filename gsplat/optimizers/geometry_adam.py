# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused activation + Adam step for the geometry parameters.

``GeometryAdam.activate(means, quats, log_scales, opacity_logits)`` returns
``(means, quats, exp(log_scales), sigmoid(opacity_logits))``. Its backward does
not produce gradients for the four parameters; instead one CUDA kernel applies
the activation backward and the Adam step of each parameter's
``torch.optim.Adam`` in place (their ``.grad`` stays ``None``, so the usual
``optimizer.step()`` skips them). Adam state stays in the torch optimizers, so
densification utilities keep working. Call ``activate`` once per optimizer step.
"""

import math
from typing import Dict

import torch
from torch import Tensor

from ..cuda._wrapper import _make_lazy_cuda_func


class GeometryAdam:
    NAMES = ("means", "quats", "scales", "opacities")

    def __init__(self, optimizers: Dict[str, torch.optim.Adam]):
        self.opts = [optimizers[n] for n in self.NAMES]
        # Extra means gradient handed over by another backward (e.g. LazySHAdam)
        # instead of going through autograd's accumulation; added in-kernel.
        self.extra_means_grad = None

    def add_means_grad(self, g: Tensor):
        if self.extra_means_grad is None:
            self.extra_means_grad = g
        else:
            self.extra_means_grad = self.extra_means_grad + g

    def _state(self, opt):
        (group,) = opt.param_groups
        (p,) = group["params"]
        assert group.get("weight_decay", 0) == 0 and not group.get("amsgrad", False)
        assert not group.get("maximize", False)
        state = opt.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return group, p, state

    def _hp(self, group, state):
        state["step"] += 1
        step = float(state["step"])
        beta1, beta2 = group["betas"]
        return [beta1, beta2, group["eps"], group["lr"] / (1 - beta1**step), math.sqrt(1 - beta2**step)]

    def activate(self, means: Tensor, quats: Tensor, log_scales: Tensor, opacity_logits: Tensor):
        if not torch.is_grad_enabled():
            return means, quats, torch.exp(log_scales), torch.sigmoid(opacity_logits)
        return _GeomActivate.apply(self, means, quats, log_scales, opacity_logits)

    def step(self, g_means, g_quats, g_scales, scales_act, g_opac, opac_act):
        args = []
        for opt, g, act in zip(self.opts, (g_means, g_quats, g_scales, g_opac), (None, None, scales_act, opac_act)):
            group, p, state = self._state(opt)
            args += [p.data, state["exp_avg"], state["exp_avg_sq"], None if g is None else g.contiguous()]
            has_grad = g is not None
            if opt is self.opts[0]:
                extra, self.extra_means_grad = self.extra_means_grad, None
                args.append(None if extra is None else extra.contiguous())
                has_grad = has_grad or extra is not None
            if act is not None:
                args.append(act)
            # Like torch.optim.Adam: a parameter without gradient is skipped and
            # its step count does not advance (the kernel skips it too).
            args.append(self._hp(group, state) if has_grad else [0.9, 0.999, 1e-15, 0.0, 1.0])
        _make_lazy_cuda_func("geometry_adam")(*args)


class _GeomActivate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mgr, means, quats, log_scales, opacity_logits):
        scales = torch.exp(log_scales)
        opac = torch.sigmoid(opacity_logits)
        ctx.mgr = mgr
        ctx.save_for_backward(scales, opac)
        return means.view_as(means), quats.view_as(quats), scales, opac

    @staticmethod
    def backward(ctx, g_means, g_quats, g_scales, g_opac):
        scales, opac = ctx.saved_tensors
        ctx.mgr.step(g_means, g_quats, g_scales, scales, g_opac, opac)
        return None, None, None, None, None
