# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases of the fused SH kernels (boundary values, channel-count guards)."""
import pytest
import torch
from gsplat.cuda._wrapper import _make_lazy_cuda_func

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def inputs(channels=3):
    means = torch.tensor([[.1, .2, 2.], [.4, -.3, 3.]], device='cuda')
    vm = torch.eye(4, device='cuda')[None]
    sh0 = torch.full((2, 1, channels), .1, device='cuda')
    shn = torch.zeros((2, 3, channels), device='cuda')
    return means, vm, sh0, shn


def fused_step(means, vm, sh0, shn, grad, degree=1):
    sh0, shn = sh0.clone(), shn.clone()
    mn, vn = torch.zeros_like(shn), torch.zeros_like(shn)
    m0, v0 = torch.zeros_like(sh0), torch.zeros_like(sh0)
    vmeans = _make_lazy_cuda_func('sh_colors_fused_bwd_adam')(
        degree, means, vm, sh0, shn, grad,
        mn, vn, .01, .9, .999, 1e-8, .03162277660168379,
        m0, v0, .01, .9, .999, 1e-8, .03162277660168379, True,
    )
    torch.cuda.synchronize()
    return sh0, shn, m0, v0, mn, vn, vmeans


def test_direct_eager_row_strided_colors_grad():
    means, vm, sh0, shn = inputs()
    packed = torch.full((1, 2, 9), -23., device='cuda')
    packed[..., :3] = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], device='cuda')
    view = packed[..., :3]
    assert not view.is_contiguous()
    actual = fused_step(means, vm, sh0, shn, view)
    expected = fused_step(means, vm, sh0, shn, view.contiguous())
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_clamp_boundary_matches_installed_torch():
    means, vm, sh0, shn = inputs()
    c0 = torch.tensor(.2820947917738781, device='cuda', dtype=torch.float32)
    sh0.fill_((-torch.tensor(.5, device='cuda') / c0).item())
    raw = sh0 * c0 + .5
    assert torch.equal(raw, torch.zeros_like(raw)), 'need exact zero preactivation'
    ref = sh0.clone().requires_grad_()
    (ref * c0 + .5).clamp_min(0).sum().backward()
    actual_color = _make_lazy_cuda_func('sh_colors_fused_fwd')(0, means, vm, sh0, shn)
    torch.testing.assert_close(actual_color, torch.zeros_like(actual_color), atol=0, rtol=0)
    result = fused_step(means, vm, sh0, shn, torch.ones((1,2,3), device='cuda'), degree=0)
    # First Adam moment = (1-beta1)*gradient. Match this torch installation's
    # boundary convention rather than assuming the derivative at zero.
    torch.testing.assert_close(result[2], ref.grad * (1-torch.tensor(.9,device='cuda')), atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize('degree', [0, 1])
def test_l1_eager_rejects_channels_outside_scratch_capacity(degree):
    means, vm, _, shn = inputs(channels=5)
    with pytest.raises(RuntimeError, match='channel|1..4'):
        _make_lazy_cuda_func('spherical_harmonics_l1_plus_bwd_adam')(
            degree, means, vm, shn, torch.ones((1,2,5),device='cuda'),
            torch.zeros_like(shn), torch.zeros_like(shn),
            .01, .9, .999, 1e-8, .03162277660168379, False,
        )
