# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the fused training kernels against the plain PyTorch / unfused reference.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math

import pytest
import torch

device = torch.device("cuda:0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")

LRS = dict(means=1.6e-4, quats=1e-3, scales=5e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
BETAS, EPS = (0.9, 0.999), 1e-15


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def make_splats(N, seed=0):
    g = torch.Generator(device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, generator=g)  # noqa: E731
    return dict(
        means=r(N, 3) * torch.tensor([2.0, 1.5, 1.0], device=device) + torch.tensor([0, 0, 4.0], device=device),
        quats=r(N, 4),
        scales=torch.rand(N, 3, device=device, generator=g) * 3 - 6,
        opacities=r(N),
        sh0=r(N, 1, 3) * 0.5,
        shN=r(N, 15, 3) * 0.1,
    )


def adam(p, lr, fused=True):
    return torch.optim.Adam([p], lr=lr, eps=EPS, betas=BETAS, fused=fused)


def view_matrix(step, C=1):
    ang = 0.05 * step
    vm = torch.eye(4, device=device).repeat(C, 1, 1)
    vm[:, :3, :3] = torch.tensor(
        [[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]], device=device
    )
    vm[:, 0, 3] = 0.1 * step
    return vm


def intrinsics(W, H, C=1):
    return torch.tensor([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]], device=device).repeat(C, 1, 1)


def adam_state(opt, p):
    return [opt.state[p][k] for k in ("exp_avg", "exp_avg_sq")]


def test_geometry_adam():
    from gsplat.optimizers.geometry_adam import GeometryAdam

    torch.manual_seed(0)
    init = {k: v for k, v in make_splats(100_000).items() if k in ("means", "quats", "scales", "opacities")}
    P = {k: {n: torch.nn.Parameter(v.clone()) for n, v in init.items()} for k in ("ref", "fus")}
    O = {k: {n: adam(P[k][n], LRS[n]) for n in init} for k in P}
    geo = GeometryAdam(O["fus"])
    for _ in range(10):
        w = {
            n: torch.randn_like(v) * (torch.rand(v.shape[0], *([1] * (v.dim() - 1)), device=device) < 0.4)
            for n, v in init.items()
        }
        for k, p in P.items():
            if k == "ref":
                outs = p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"])
            else:
                outs = geo.activate(p["means"], p["quats"], p["scales"], p["opacities"])
            sum(
                (o * w[n]).sum() for o, n in zip(outs, ("means", "quats", "scales", "opacities"))
            ).backward()
            for opt in O[k].values():
                opt.step()
                opt.zero_grad(set_to_none=True)
        for n in init:
            assert rel(P["fus"][n].data, P["ref"][n].data) < 1e-5, n
            for a, b in zip(adam_state(O["fus"][n], P["fus"][n]), adam_state(O["ref"][n], P["ref"][n])):
                assert rel(a, b) < 1e-5, n


def _geo_pair(N):
    from gsplat.optimizers.geometry_adam import GeometryAdam

    init = {k: v for k, v in make_splats(N).items() if k in ("means", "quats", "scales", "opacities")}
    P = {k: {n: torch.nn.Parameter(v.clone()) for n, v in init.items()} for k in ("ref", "fus")}
    O = {k: {n: torch.optim.Adam([P[k][n]], lr=LRS[n], eps=EPS) for n in init} for k in P}
    return init, P, O, {k: GeometryAdam(O[k]) for k in P}


def _assert_same(init, P, O, tol=0.0):
    for n in init:
        pairs = [(P["fus"][n].data, P["ref"][n].data)]
        pairs += list(zip(adam_state(O["fus"][n], P["fus"][n]), adam_state(O["ref"][n], P["ref"][n])))
        for a, b in pairs:
            assert float((a - b).abs().max()) <= tol, n
        assert float(O["fus"][n].state[P["fus"][n]]["step"]) == float(O["ref"][n].state[P["ref"][n]]["step"])


def test_projection_adam():
    from gsplat.cuda._wrapper import fully_fused_projection

    torch.manual_seed(0)
    N, W, H = 100_000, 640, 480
    init, P, O, G = _geo_pair(N)
    K = intrinsics(W, H)
    for step in range(12):
        vm = view_matrix(step)
        w2, w3, wo = torch.randn(1, N, 2, device=device), torch.randn(1, N, 3, device=device), torch.randn(1, N, device=device)
        wd = torch.randn(1, N, device=device) if step % 2 else None
        extra = torch.randn(N, 3, device=device) * (torch.rand(N, 1, device=device) < 0.3) if step % 3 else None
        fwd = {}
        for k, p in P.items():
            if extra is not None:
                G[k].add_means_grad(extra.clone())
            if k == "ref":
                m, q, s, o = G[k].activate(p["means"], p["quats"], p["scales"], p["opacities"])
                radii, means2d, depths, conics, _ = fully_fused_projection(m, None, q, s, vm, K, W, H, opacities=o)
            else:
                m, o, radii, means2d, depths, conics, _ = G[k].project(
                    p["means"], p["quats"], p["scales"], p["opacities"], vm, K, W, H
                )
            fwd[k] = [radii, means2d, depths, conics, o]
            loss = (means2d * w2).sum() + (conics * w3).sum() + (o[None] * wo).sum()
            if wd is not None:
                loss = loss + (depths * wd).sum()
            loss.backward()
        msk = fwd["ref"][0][..., 0] > 0  # outputs of culled gaussians are unset
        assert msk.any()
        assert torch.equal(fwd["fus"][0][..., 0] > 0, msk)
        for name, a, b in zip(("radii", "means2d", "depths", "conics", "opac"), fwd["fus"], fwd["ref"]):
            assert torch.equal(a, b) if name == "opac" else torch.equal(a[msk], b[msk]), name
        _assert_same(init, P, O)


def test_packed_stats():
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    torch.manual_seed(0)
    N, W, H = 100_000, 640, 480
    init, P, O, G = _geo_pair(N)
    S = {k: (torch.zeros(N, device=device), torch.zeros(N, device=device)) for k in P}
    K = intrinsics(W, H)
    for step in range(10):
        vm = view_matrix(step)
        for k, p in P.items():
            radii = G[k].project(p["means"], p["quats"], p["scales"], p["opacities"], vm, K, W, H)[2]
            vis = (radii[..., 0] > 0) & (radii[..., 1] > 0)
            g = torch.Generator(device=device).manual_seed(step)
            packed = torch.randn(1, N, 9, device=device, generator=g) * vis[..., None]
            v_conics, v_means2d, v_opac = packed[..., 3:6], packed[..., 6:8], packed[..., 8]
            _, o, radii, means2d, _, conics, _ = G[k].project(p["means"], p["quats"], p["scales"], p["opacities"], vm, K, W, H)
            if k == "fus":
                G[k].strategy_stats = (S[k][0], S[k][1], W / 2.0, H / 2.0)
                torch.autograd.backward([means2d, conics, o], [v_means2d, v_conics, v_opac[0]])
            else:
                torch.autograd.backward(
                    [means2d, conics, o], [v_means2d.contiguous(), v_conics.contiguous(), v_opac[0].contiguous()]
                )
                _make_lazy_cuda_func("strategy_update_state")(
                    v_means2d.contiguous(), radii.contiguous(), W / 2.0, H / 2.0, S[k][0], S[k][1]
                )
        _assert_same(init, P, O)
        for a, b in zip(S["fus"], S["ref"]):
            assert torch.equal(a, b)
    assert float(S["ref"][0].sum()) > 0


def _sh_setup(N=20_000, D=3):
    sp = make_splats(N)
    for k in ("sh0", "shN"):
        sp[k] = sp[k][..., :D] if D <= 3 else torch.cat([sp[k], sp[k][..., : D - 3]], -1)
    return sp


def test_fused_sh_adam():
    from gsplat import spherical_harmonics_l1_plus
    from gsplat.optimizers.fused_sh_adam import spherical_harmonics_l1_plus_fused_adam

    torch.manual_seed(0)
    sp = _sh_setup()
    N = sp["means"].shape[0]
    ref, fus = (torch.nn.Parameter(sp["shN"].clone()) for _ in range(2))
    m_ref, m_fus = (sp["means"].clone().requires_grad_(True) for _ in range(2))
    opt_ref, opt_fus = adam(ref, LRS["shN"]), adam(fus, LRS["shN"])
    for degree in [0, 1, 1, 2, 3, 3, 3, 3, 3, 3]:
        vm = torch.eye(4, device=device)[None].clone()
        vm[0, :3, 3] = torch.randn(3, device=device)
        v_colors = torch.randn(1, N, 3, device=device) * (torch.rand(1, N, 1, device=device) < 0.4)
        spherical_harmonics_l1_plus(degree, m_ref, vm, ref).backward(v_colors)
        opt_ref.step()
        opt_ref.zero_grad(set_to_none=True)
        spherical_harmonics_l1_plus_fused_adam(degree, m_fus, vm, fus, opt_fus).backward(v_colors)
        assert fus.grad is None
        opt_fus.step()  # no-op: grad is None
        assert rel(fus.data, ref.data) < 1e-5
        for a, b in zip(adam_state(opt_fus, fus), adam_state(opt_ref, ref)):
            assert rel(a, b) < 1e-5
        if m_ref.grad is not None and m_ref.grad.norm() > 0:
            assert rel(m_fus.grad, m_ref.grad) < 1e-5
        m_ref.grad = m_fus.grad = None
    assert float(opt_ref.state[ref]["step"]) == float(opt_fus.state[fus]["step"])


def test_fused_colors():
    from gsplat import spherical_harmonics_l0, spherical_harmonics_l1_plus
    from gsplat.optimizers.fused_sh_adam import sh_colors_fused_adam

    torch.manual_seed(0)
    sp = _sh_setup()
    N = sp["means"].shape[0]
    P = {k: {n: torch.nn.Parameter(sp[n].clone()) for n in ("sh0", "shN")} for k in ("ref", "fus")}
    M = {k: sp["means"].clone().requires_grad_(True) for k in P}
    O = {k: {n: adam(P[k][n], LRS[n]) for n in ("sh0", "shN")} for k in P}
    for step, degree in enumerate([0, 0, 1, 2, 3, 3, 3, 3, 3, 3]):
        vm = torch.eye(4, device=device)[None].clone()
        vm[0, :3, 3] = torch.randn(3, device=device)
        v_colors = torch.randn(1, N, 3, device=device) * (torch.rand(1, N, 1, device=device) < 0.4)
        if step > 0:  # compare every step from identical state (clamp masks flip chaotically otherwise)
            with torch.no_grad():
                for n in ("sh0", "shN"):
                    P["fus"][n].copy_(P["ref"][n])
                    for a, b in zip(adam_state(O["fus"][n], P["fus"][n]), adam_state(O["ref"][n], P["ref"][n])):
                        a.copy_(b)
        p = P["ref"]
        col_ref = torch.clamp_min(
            spherical_harmonics_l0(p["sh0"]) + spherical_harmonics_l1_plus(degree, M["ref"], vm, p["shN"]) + 0.5, 0.0
        )
        col_ref.backward(v_colors)
        for o in O["ref"].values():
            o.step()
            o.zero_grad(set_to_none=True)
        p = P["fus"]
        col_fus = sh_colors_fused_adam(degree, M["fus"], vm, p["sh0"], p["shN"], O["fus"]["sh0"], O["fus"]["shN"])
        col_fus.backward(v_colors)
        assert p["sh0"].grad is None and p["shN"].grad is None
        assert float((col_fus - col_ref).abs().max()) < 1e-5
        for n in ("sh0", "shN"):
            assert rel(P["fus"][n].data, P["ref"][n].data) < 1e-5, n
            for a, b in zip(adam_state(O["fus"][n], P["fus"][n]), adam_state(O["ref"][n], P["ref"][n])):
                assert rel(a, b) < 1e-5, n
        if M["ref"].grad is not None and M["ref"].grad.norm() > 0:
            assert rel(M["fus"].grad, M["ref"].grad) < 1e-5
        M["ref"].grad = M["fus"].grad = None


@pytest.mark.parametrize("D,frac", [(3, 0.4), (1, 0.4), (4, 0.4), (3, 0.9)])
def test_lazy_sh_adam(D, frac):
    from gsplat.optimizers.fused_sh_adam import LazySHAdam, sh_colors_fused_adam

    torch.manual_seed(0)
    sp = _sh_setup(D=D)
    N = sp["means"].shape[0]
    E, L = ({n: torch.nn.Parameter(sp[n].clone()) for n in ("sh0", "shN")} for _ in range(2))
    OE, OL = ({n: adam(P[n], LRS[n]) for n in P} for P in (E, L))
    lazy = LazySHAdam(OL["sh0"], OL["shN"])
    me, ml = (sp["means"].clone().requires_grad_(True) for _ in range(2))
    for step, degree in enumerate([0] * 5 + [1] * 5 + [2] * 5 + [3] * 25, 1):
        vm = torch.eye(4, device=device)[None].clone()
        vm[0, :3, 3] = torch.randn(3, device=device) * 0.3
        mask = torch.rand(1, N, device=device) < frac
        v_colors = torch.randn(1, N, D, device=device) * mask[..., None]
        ce = sh_colors_fused_adam(degree, me, vm, E["sh0"], E["shN"], OE["sh0"], OE["shN"])
        cl = lazy.colors(degree, ml, vm, mask)
        assert torch.equal(ce[mask], cl[mask])
        ce.backward(v_colors)
        cl.backward(v_colors)
        assert float((me.grad - ml.grad).abs().max()) < 1e-4  # atomics: float-order noise only
        me.grad = ml.grad = None
        if step % 10 == 0:
            lazy.catch_up()
    lazy.catch_up()
    for n in ("sh0", "shN"):
        assert torch.equal(E[n].data, L[n].data), n
        for a, b in zip(adam_state(OE[n], E[n]), adam_state(OL[n], L[n])):
            assert torch.equal(a, b), n
        assert float(OE[n].state[E[n]]["step"]) == float(OL[n].state[L[n]]["step"])


def test_means_handoff():
    from gsplat.optimizers.fused_sh_adam import LazySHAdam
    from gsplat.optimizers.geometry_adam import GeometryAdam

    sp = make_splats(20_000)
    res = {}
    for sink in (False, True):
        torch.manual_seed(0)
        P = {k: torch.nn.Parameter(v.clone()) for k, v in sp.items()}
        O = {k: adam(P[k], LRS[k]) for k in LRS}
        lazy, geo = LazySHAdam(O["sh0"], O["shN"]), GeometryAdam(O)
        if sink:
            lazy.means_grad_sink = geo
        N = P["means"].shape[0]
        for _ in range(5):
            vm = torch.eye(4, device=device)[None].clone()
            vm[0, :3, 3] = torch.randn(3, device=device)
            mask = torch.rand(1, N, device=device) < 0.4
            m, q, s, o = geo.activate(P["means"], P["quats"], P["scales"], P["opacities"])
            colors = lazy.colors(3, m, vm, mask)
            w = torch.randn(N, 3, device=device)
            loss = (colors * torch.randn_like(colors) * mask[..., None]).sum() + (m * w).sum() + q.sum() + s.sum() + o.sum()
            loss.backward()
            for opt in O.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
        lazy.catch_up()
        res[sink] = {k: v.detach().clone() for k, v in P.items()}
    for k in res[False]:
        assert torch.equal(res[False][k], res[True][k]), k


def _isect_scene(N, W, H, tile, C=1, seed=0, gaussians_only_2d=True):
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles

    g = torch.Generator(device).manual_seed(seed)
    means = torch.randn(N, 3, device=device, generator=g) * torch.tensor([1.5, 1.0, 0.5], device=device) + torch.tensor(
        [0, 0, 4.0], device=device
    )
    quats = torch.randn(N, 4, device=device, generator=g)
    scales = torch.rand(N, 3, device=device, generator=g) * 0.08 + 0.005
    opac = torch.rand(N, device=device, generator=g) * 0.95 + 0.04
    vm = torch.eye(4, device=device).repeat(C, 1, 1)
    vm[:, 0, 3] = torch.linspace(-0.2, 0.2, C, device=device)
    K = torch.tensor([[180.0, 0, W / 2], [0, 180.0, H / 2], [0, 0, 1]], device=device).repeat(C, 1, 1)
    radii, means2d, depths, conics, _ = fully_fused_projection(means, None, quats, scales, vm, K, W, H, opacities=opac)
    o = opac[None].expand(C, N).contiguous()
    tw, th = math.ceil(W / tile), math.ceil(H / tile)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile, tw, th, n_images=C, conics=conics, opacities=o)
    return dict(
        means2d=means2d, conics=conics, o=o, colors=torch.rand(C, N, 3, device=device, generator=g),
        offsets=isect_offset_encode(isect_ids, C, tw, th), flat=flatten_ids, g=g, tw=tw, th=th,
    )


def _raster_reference(means2d, conics, colors, opacities, W, H, tile, offsets, flat, backgrounds=None):
    """Per-tile front-to-back compositing in PyTorch, with the CUDA kernels' cut-offs
    (alpha <= 0.999, alpha >= 1/255, stop when T <= 1e-4)."""
    C, N, D = colors.shape
    th, tw = offsets.shape[1:]
    starts = offsets.flatten().tolist()
    ends = starts[1:] + [flat.numel()]
    m, cn, col, op = means2d.reshape(-1, 2), conics.reshape(-1, 3), colors.reshape(-1, D), opacities.reshape(-1)
    img, alpha_out = colors.new_zeros(C, H, W, D), colors.new_zeros(C, H, W, 1)
    for t, (s, e) in enumerate(zip(starts, ends)):
        c, y0, x0 = t // (th * tw), (t // tw) % th * tile, t % tw * tile
        y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
        ys, xs = torch.meshgrid(torch.arange(y0, y1, device=device) + 0.5, torch.arange(x0, x1, device=device) + 0.5, indexing="ij")
        ids = flat[s:e]
        d = torch.stack([xs, ys], -1).reshape(-1, 1, 2) - m[ids]
        a, b, cc = cn[ids].unbind(-1)
        sigma = 0.5 * (a * d[..., 0] ** 2 + cc * d[..., 1] ** 2) + b * d[..., 0] * d[..., 1]
        al = torch.clamp_max(op[ids] * torch.exp(-sigma), 0.999)
        al = torch.where((sigma >= 0) & (al >= 1 / 255), al, torch.zeros_like(al))
        T = torch.cumprod(torch.cat([al.new_ones(len(d), 1), 1 - al], 1), 1)
        w = torch.where(T[:, 1:] > 1e-4, al * T[:, :-1], torch.zeros_like(al))
        acc = w.sum(1, keepdim=True)
        px = w @ col[ids]
        if backgrounds is not None:
            px = px + (1 - acc) * backgrounds[c]
        img[c, y0:y1, x0:x1] = px.reshape(y1 - y0, x1 - x0, D)
        alpha_out[c, y0:y1, x0:x1] = acc.reshape(y1 - y0, x1 - x0, 1)
    return img, alpha_out


def _raster_grads(sc, W, H, tile, ch=3, bg=False, absgrad=False, fn=None):
    from gsplat.cuda._wrapper import rasterize_to_pixels

    fn = fn or rasterize_to_pixels
    g = torch.Generator(device).manual_seed(1)
    C, N = sc["o"].shape
    colors = torch.rand(C, N, ch, device=device, generator=g)
    backgrounds = torch.rand(C, ch, device=device, generator=g) if bg else None
    leaves = [sc["means2d"].detach(), sc["conics"].detach(), colors, sc["o"].detach()]
    if bg:
        leaves.append(backgrounds)
    leaves = [t.clone().requires_grad_(True) for t in leaves]
    kw = dict(absgrad=True) if absgrad else {}
    img, alpha = fn(leaves[0], leaves[1], leaves[2], leaves[3], W, H, tile, sc["offsets"], sc["flat"],
                    backgrounds=leaves[4] if bg else None, **kw)
    w = torch.rand(img.shape, device=device, generator=g)
    wa = torch.rand(alpha.shape, device=device, generator=g)
    grads = torch.autograd.grad((img * w).sum() + (alpha * wa).sum(), leaves)
    out = dict(img=img.detach(), alpha=alpha.detach(), **{f"g{i}": gr for i, gr in enumerate(grads)})
    if absgrad:
        out["absgrad"] = leaves[0].absgrad
    return out


def test_tile4_vs_tile16():
    torch.manual_seed(0)
    W, H, N = 64, 48, 400
    g = torch.Generator(device).manual_seed(0)
    means2d = torch.rand(1, N, 2, device=device, generator=g) * torch.tensor([W, H], device=device)
    sig = torch.rand(1, N, device=device, generator=g) * 3 + 1
    conics = torch.stack([1 / sig**2, torch.zeros_like(sig), 1 / sig**2], -1)
    radii = torch.ceil(4 * sig).int()[..., None].repeat(1, 1, 2)
    depths = torch.rand(1, N, device=device, generator=g)
    colors = torch.rand(1, N, 3, device=device, generator=g)
    opac = torch.rand(1, N, device=device, generator=g) * 0.9

    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, rasterize_to_pixels

    def run(ts):
        tw, th = math.ceil(W / ts), math.ceil(H / ts)
        _, ids, flat = isect_tiles(means2d, radii, depths, ts, tw, th)
        offs = isect_offset_encode(ids, 1, tw, th)
        m, c, col, o = [t.clone().requires_grad_(True) for t in (means2d, conics, colors, opac)]
        img, _ = rasterize_to_pixels(m, c, col, o, W, H, ts, offs, flat)
        (img * torch.linspace(0, 1, img.numel(), device=device).view_as(img)).sum().backward()
        return img.detach(), [t.grad for t in (m, c, col, o)]

    i16, g16 = run(16)
    i4, g4 = run(4)
    assert torch.allclose(i16, i4, atol=1e-5)
    for a, b in zip(g16, g4):
        assert rel(a, b) < 2e-3


@pytest.mark.parametrize("tile", [16, 4])
@pytest.mark.parametrize("C,ch,bg", [(1, 3, False), (2, 3, True), (1, 1, True), (2, 4, False), (1, 5, True), (2, 8, False)])
def test_raster_vs_torch_reference(tile, C, ch, bg):
    W, H = 97, 65
    sc = _isect_scene(1500, W, H, tile, C=C, seed=C * 10 + ch)
    out = _raster_grads(sc, W, H, tile, ch=ch, bg=bg)
    ref = _raster_grads(sc, W, H, tile, ch=ch, bg=bg, fn=_raster_reference)
    for k, v in out.items():
        assert rel(v, ref[k]) < (1e-4 if k in ("img", "alpha") else 2e-3), k


@pytest.mark.parametrize("ch,bg,absgrad", [(3, False, True), (3, True, True), (1, False, False), (4, True, True), (8, False, False)])
def test_raster_tile4_matches_tile16(ch, bg, absgrad):
    W, H = 197, 131
    outs = [_raster_grads(_isect_scene(2000, W, H, t, C=2, seed=ch), W, H, t, ch=ch, bg=bg, absgrad=absgrad) for t in (16, 4)]
    for k in outs[0]:
        assert rel(outs[1][k], outs[0][k]) < (1e-4 if k in ("img", "alpha") else 2e-3), k


@pytest.mark.parametrize("C,N,scale2d", [(1, 200_000, 0), (3, 50_000, 0), (2, 50_000, 0.01)])
def test_strategy_state(C, N, scale2d):
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(0)
    g = torch.randn(C, N, 2, device=device) * 1e-4
    r = torch.randint(0, 8, (C, N, 2), device=device, dtype=torch.int32) * (torch.rand(C, N, 1, device=device) < 0.4)
    m2d = torch.zeros(C, N, 2, device=device, requires_grad=True)
    m2d.grad = g
    info = dict(width=1297, height=840, n_cameras=C, radii=r, gaussian_ids=None, means2d=m2d)
    params = {"means": torch.zeros(N, 3, device=device)}
    init = dict(
        grad2d=torch.rand(N, device=device),
        count=torch.rand(N, device=device) * 5,
        radii=torch.rand(N, device=device) * 0.01 if scale2d else None,
    )
    res = []
    for fused in (True, False):
        st = DefaultStrategy(refine_scale2d_stop_iter=1000 if scale2d else 0, fused_state_update=fused)
        state = st.initialize_state()
        state.update({k: (v.clone() if v is not None else None) for k, v in init.items()})
        i = dict(info)
        st._update_state(params, state, i)
        res.append(state)
    for k in ("grad2d", "count") + (("radii",) if scale2d else ()):
        assert torch.equal(res[0][k], res[1][k]), k


@pytest.mark.parametrize("tile", [16, 8])
def test_isect_counted(tile):
    from gsplat.cuda._wrapper import isect_tiles

    if not hasattr(torch.ops.gsplat, "intersect_tile_counted"):
        pytest.skip("intersect_tile_counted not available")
    torch.manual_seed(0)
    N, W, H = 100_000, 640, 480
    sp = make_splats(N)
    K = intrinsics(W, H)
    for step in range(0, 12, 3):
        vm = view_matrix(step)
        tw, th = math.ceil(W / tile), math.ceil(H / tile)
        radii, m2d, depths, conics, opac, tiles = torch.ops.gsplat.projection_act_fwd(
            sp["means"], sp["quats"], sp["scales"], sp["opacities"], vm, K.contiguous(), W, H, 0.3, 0.01, 1e10, tile, tw, th
        )
        o = opac[None]
        ref_tiles, ref_ids, ref_flat = isect_tiles(m2d, radii, depths, tile, tw, th, n_images=1, conics=conics, opacities=o)
        ids, flat = torch.ops.gsplat.intersect_tile_counted(m2d, radii, depths, conics, o, tiles, tile, tw, th)
        assert ref_ids.numel() > 0
        assert torch.equal(tiles, ref_tiles)
        assert torch.equal(ids, ref_ids) and torch.equal(flat, ref_flat)


def test_fused_l1_ssim():
    import torch.nn.functional as F

    fused_ssim = pytest.importorskip("fused_ssim")
    if not hasattr(fused_ssim, "fused_l1_ssim"):
        pytest.skip("fused_ssim.fused_l1_ssim not available")
    torch.manual_seed(0)
    H, W = 168, 259
    gt = torch.rand(1, H, W, 3, device=device)
    base = torch.rand(1, H, W, 3, device=device)
    base[:, :10, :10] = gt[:, :10, :10]  # exact zeros in the difference
    out = {}
    for fused in (False, True):
        x = base.clone().requires_grad_(True)
        a, b = x.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)
        if fused:
            l1, s = fused_ssim.fused_l1_ssim(a, b, padding="valid")
        else:
            l1, s = F.l1_loss(x, gt, reduction="none").mean(), fused_ssim.fused_ssim(a, b, padding="valid")
        torch.lerp(l1, 1.0 - s, 0.2).backward()
        out[fused] = (float(l1), float(s), x.grad.clone())
    assert abs(out[False][0] - out[True][0]) < 1e-6 and abs(out[False][1] - out[True][1]) < 1e-6
    assert torch.equal(out[False][2], out[True][2])
