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
        v_means = _make_lazy_cuda_func("spherical_harmonics_l1_plus_bwd_adam")(
            ctx.degree,
            means.contiguous(),
            viewmats.contiguous(),
            shN.data,
            v_colors.contiguous(),
            *_adam_args(ctx.optimizer, shN),
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


def _adam_args(opt: torch.optim.Adam, param: Tensor):
    """Advance ``opt``'s step for ``param`` and return its bias-corrected Adam
    hyper-parameters and state, initializing the state like torch.optim.Adam."""
    (group,) = opt.param_groups
    (p,) = group["params"]
    assert p is param, "optimizer must own exactly this parameter"
    assert group.get("weight_decay", 0) == 0 and not group.get("amsgrad", False)
    assert not group.get("maximize", False)
    state = opt.state[p]
    if len(state) == 0:
        state["step"] = torch.tensor(0.0, dtype=torch.float32)
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    state["step"] += 1
    step = float(state["step"])
    beta1, beta2 = group["betas"]
    return (
        state["exp_avg"],
        state["exp_avg_sq"],
        group["lr"] / (1 - beta1**step),
        beta1,
        beta2,
        group["eps"],
        math.sqrt(1 - beta2**step),
    )


class _SHColorsFusedAdam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, degree, means, viewmats, sh0, shN, opt_sh0, opt_shN):
        ctx.degree = degree
        ctx.opts = (opt_sh0, opt_shN)
        ctx.save_for_backward(means, viewmats, sh0, shN)
        return _make_lazy_cuda_func("sh_colors_fused_fwd")(degree, means.contiguous(), viewmats, sh0, shN)

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        means, viewmats, sh0, shN = ctx.saved_tensors
        opt_sh0, opt_shN = ctx.opts
        v_means = _make_lazy_cuda_func("sh_colors_fused_bwd_adam")(
            ctx.degree,
            means.contiguous(),
            viewmats.contiguous(),
            sh0.data,
            shN.data,
            v_colors.contiguous(),
            *_adam_args(opt_shN, shN),
            *_adam_args(opt_sh0, sh0),
            ctx.needs_input_grad[1],
        )
        return None, v_means, None, None, None, None, None


def sh_colors_fused_adam(
    degree: int,
    means: Tensor,
    viewmats: Tensor,
    sh0: Tensor,
    shN: Tensor,
    opt_sh0: torch.optim.Adam,
    opt_shN: torch.optim.Adam,
) -> Tensor:
    """``max(SH(sh0, shN, dir) + 0.5, 0)`` per camera, ``[C, N, D]`` (dense,
    unbatched), bitwise equal to ``spherical_harmonics_l0`` +
    ``spherical_harmonics_l1_plus`` followed by ``(+ 0.5).clamp_min(0)``.

    The backward pass performs the Adam steps of ``opt_sh0`` and ``opt_shN`` in
    place instead of producing gradients for ``sh0`` / ``shN`` (their ``.grad``
    stays ``None``). Without grad mode it is a plain fused forward.
    """
    if not torch.is_grad_enabled():
        return _make_lazy_cuda_func("sh_colors_fused_fwd")(degree, means.contiguous(), viewmats.contiguous(), sh0, shN)
    return _SHColorsFusedAdam.apply(degree, means, viewmats.contiguous(), sh0, shN, opt_sh0, opt_shN)


class LazySHAdam:
    """Exact lazy Adam for ``sh0`` / ``shN`` with fused SH color evaluation.

    A zero-gradient Adam step is deterministic and SH coefficients do not affect
    visibility, so rows without color gradient skip their update; the missed
    steps are replayed (bitwise identically) when the row is next needed: in the
    forward for visible rows, in the backward before a row's next real update,
    and for all rows in :meth:`catch_up`. Call :meth:`catch_up` before anything
    else reads or edits ``sh0``/``shN`` or their Adam state (densification,
    evaluation, checkpointing) and at the end of training.

    Uses the given ``torch.optim.Adam`` instances only as containers for the
    hyper-parameters and the ``exp_avg`` / ``exp_avg_sq`` / ``step`` state.
    """

    _CHUNK = 1024

    def __init__(self, opt_sh0: torch.optim.Adam, opt_shN: torch.optim.Adam):
        self.opts = (opt_sh0, opt_shN)
        self.step = 0  # optimizer steps applied (eagerly or deferred)
        self.last: Tensor = None  # [N] int32, last step applied per row
        self.caught_up_at = 0
        self.tabs: Tensor = None  # [4, T] float32
        # Optional GeometryAdam (or anything with add_means_grad) that takes the
        # SH contribution to the means gradient directly.
        self.means_grad_sink = None
        self._tab_filled = 0  # steps 1.._tab_filled are valid
        self._tab_hparams = None

    def _param_state(self, opt):
        (group,) = opt.param_groups
        (p,) = group["params"]
        assert group.get("weight_decay", 0) == 0 and not group.get("amsgrad", False)
        assert not group.get("maximize", False)
        state = opt.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(float(self.step), dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return group, p, state

    def _hparams(self):
        out = []
        for opt in self.opts:
            (group,) = opt.param_groups
            out.append((float(group["lr"]), *map(float, group["betas"]), float(group["eps"])))
        return tuple(out)

    def _ensure_tables(self, step: int, device):
        hp = self._hparams()
        if hp != self._tab_hparams:
            # Hyper-parameters changed: entries from `step` on must be refilled.
            self._tab_filled = min(self._tab_filled, step - 1)
            self._tab_hparams = hp
        if step <= self._tab_filled:
            return
        lo, hi = self._tab_filled + 1, step + self._CHUNK
        if self.tabs is None or self.tabs.shape[1] <= hi:
            cap = max(hi + 1, 2 * (0 if self.tabs is None else self.tabs.shape[1]))
            new = torch.zeros(4, cap, dtype=torch.float32, device=device)
            if self.tabs is not None:
                new[:, : self.tabs.shape[1]] = self.tabs
            self.tabs = new
        rows = []
        for lr, b1, b2, _ in hp:
            rows.append([lr / (1 - b1**s) for s in range(lo, hi + 1)])
            rows.append([math.sqrt(1 - b2**s) for s in range(lo, hi + 1)])
        # Same rounding as passing the Python floats to a kernel (double -> float).
        self.tabs[:, lo : hi + 1] = torch.tensor(rows, dtype=torch.float32).to(device)
        self._tab_filled = hi

    def _ensure_last(self, n: int, device):
        if self.last is None or self.last.numel() != n:
            assert self.last is None or self.caught_up_at == self.step, (
                "sh0/shN changed size without LazySHAdam.catch_up() first"
            )
            self.last = torch.full((n,), self.step, dtype=torch.int32, device=device)

    @torch.no_grad()
    def catch_up(self):
        """Apply all deferred steps so sh0/shN and their Adam state are current."""
        if self.last is None or self.step == 0:
            self.caught_up_at = self.step
            return
        self._ensure_tables(self.step, self.last.device)
        for i, opt in enumerate(self.opts):
            group, p, state = self._param_state(opt)
            b1, b2 = group["betas"]
            _make_lazy_cuda_func("sh_lazy_catchup")(
                p.data, state["exp_avg"], state["exp_avg_sq"], self.last, self.step,
                self.tabs[2 * i], self.tabs[2 * i + 1], b1, b2, group["eps"],
            )
        self.last.fill_(self.step)
        self.caught_up_at = self.step

    def colors(self, degree: int, means: Tensor, viewmats: Tensor, mask: Tensor) -> Tensor:
        """``max(SH(sh0, shN, dir) + 0.5, 0)`` as ``[C, N, D]``. ``mask`` ([C, N]
        bool) must cover every gaussian rasterized in this step; other rows get
        color 0. In grad mode the backward applies this step's Adam updates."""
        _, sh0, _ = self._param_state(self.opts[0])
        _, shN, _ = self._param_state(self.opts[1])
        viewmats = viewmats.contiguous()
        if not torch.is_grad_enabled():
            self.catch_up()
            return _make_lazy_cuda_func("sh_colors_fused_fwd")(degree, means.contiguous(), viewmats, sh0, shN)
        return _LazySHColors.apply(self, degree, means, viewmats, mask.contiguous(), sh0, shN)

    def _args(self, step: int, device):
        self._ensure_tables(step, device)
        g0, sh0, s0 = self._param_state(self.opts[0])
        gN, shN, sN = self._param_state(self.opts[1])
        self._ensure_last(shN.shape[0], device)
        return (g0, sh0, s0, gN, shN, sN)


class _LazySHColors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mgr: LazySHAdam, degree, means, viewmats, mask, sh0, shN):
        step = mgr.step + 1
        g0, sh0_, s0, gN, shN_, sN = mgr._args(step, means.device)
        assert sh0_ is sh0 and shN_ is shN
        ctx.mgr, ctx.degree, ctx.step = mgr, degree, step
        ctx.save_for_backward(means, viewmats)
        return _make_lazy_cuda_func("sh_colors_lazy_fwd")(
            degree, means.contiguous(), viewmats, mask,
            sh0, s0["exp_avg"], s0["exp_avg_sq"], shN, sN["exp_avg"], sN["exp_avg_sq"],
            mgr.last, step, mgr.tabs, *g0["betas"], g0["eps"], *gN["betas"], gN["eps"],
        )

    @staticmethod
    def backward(ctx, v_colors):
        means, viewmats = ctx.saved_tensors
        mgr, step = ctx.mgr, ctx.step
        assert mgr.step == step - 1, "LazySHAdam: one backward per forward, in order"
        g0, sh0, s0, gN, shN, sN = mgr._args(step, means.device)
        v_means = _make_lazy_cuda_func("sh_colors_lazy_bwd_adam")(
            ctx.degree, means.contiguous(), viewmats, v_colors,  # row-strided views OK
            sh0.data, s0["exp_avg"], s0["exp_avg_sq"], shN.data, sN["exp_avg"], sN["exp_avg_sq"],
            mgr.last, step, mgr.tabs, *g0["betas"], g0["eps"], *gN["betas"], gN["eps"],
            ctx.needs_input_grad[2],
        )
        mgr.step = step
        s0["step"].fill_(float(step))
        sN["step"].fill_(float(step))
        if v_means is not None and mgr.means_grad_sink is not None:
            # Autograd still orders this backward before the sink's (the edge to
            # `means` exists), so the sink sees the contribution in time.
            mgr.means_grad_sink.add_means_grad(v_means)
            v_means = None
        return None, None, v_means, None, None, None, None
