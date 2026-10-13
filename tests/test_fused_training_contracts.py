# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contract checks that do not require a CUDA extension."""

import pytest
import torch

from gsplat.optimizers import fused_sh_adam as sh
from gsplat.optimizers.geometry_adam import GeometryAdam


def manager():
    p0 = torch.nn.Parameter(torch.zeros(2, 1, 3))
    pn = torch.nn.Parameter(torch.zeros(2, 3, 3))
    return sh.LazySHAdam(torch.optim.Adam([p0]), torch.optim.Adam([pn]))


def test_lazy_fallback_without_visibility_mask(monkeypatch):
    def apply(mgr, degree, means, views, mask, *params):
        assert mask.shape == (views.shape[0], means.shape[0])
        assert mask.dtype == torch.bool and mask.all()
        return mask
    monkeypatch.setattr(sh._LazySHColors, 'apply', apply)
    manager().colors(1, torch.zeros(2, 3), torch.eye(4)[None], None)


@pytest.mark.parametrize('kind', ['lazy', 'shn', 'colors', 'projection'])
def test_camera_gradients_are_rejected_before_cuda(kind):
    mgr = manager()
    means, views = torch.zeros(2, 3), torch.eye(4)[None].requires_grad_()
    p0, pn = [o.param_groups[0]['params'][0] for o in mgr.opts]
    with pytest.raises(ValueError, match='camera gradients'):
        if kind == 'lazy':
            mgr.colors(1, means, views, None)
        elif kind == 'shn':
            sh.spherical_harmonics_l1_plus_fused_adam(1, means, views, pn, mgr.opts[1])
        elif kind == 'colors':
            sh.sh_colors_fused_adam(1, means, views, p0, pn, *mgr.opts)
        else:
            GeometryAdam.__new__(GeometryAdam).project(means, None, None, None, views, torch.eye(3)[None], 16, 16)

