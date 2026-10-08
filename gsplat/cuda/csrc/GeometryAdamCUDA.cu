/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Adam for the geometry parameters (means, quats, log-scales, opacity logits)
// in one pass, with the scale (exp) and opacity (sigmoid) activation backward
// fused in. Each (parameter, element) is updated exactly like
// torch.optim.Adam (weight_decay=0, amsgrad=False) given its gradient.

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "Common.h"

namespace gsplat
{

struct GeomAdamParam
{
    float *p, *m, *v;
    const float *g; // gradient w.r.t. the activated value (nullptr: zero)
    const float *act; // activation output for the chain rule (nullptr: identity)
    float beta1, beta2, eps, step_size, bc2_sqrt;
};

enum GeomAct
{
    ACT_NONE = 0,
    ACT_EXP  = 1,
    ACT_SIGMOID = 2
};

__device__ __forceinline__ void geom_adam_elem(const GeomAdamParam &q, const int64_t i, const int act)
{
    float grad = 0.f;
    if(q.g != nullptr)
    {
        grad = q.g[i];
        if(act == ACT_EXP)
        {
            grad = __fmul_rn(grad, q.act[i]); // exp backward: grad * result
        }
        else if(act == ACT_SIGMOID)
        {
            const float y = q.act[i]; // sigmoid backward: grad * (1 - y) * y
            grad          = __fmul_rn(__fmul_rn(grad, __fsub_rn(1.f, y)), y);
        }
    }
    float p = q.p[i], m = q.m[i], v = q.v[i];
    m               = __fmaf_rn(q.beta1, m, __fmul_rn(__fsub_rn(1.f, q.beta1), grad));
    v               = __fmaf_rn(q.beta2, v, __fmul_rn(__fmul_rn(__fsub_rn(1.f, q.beta2), grad), grad));
    const float den = __fadd_rn(__fdiv_rn(__fsqrt_rn(v), q.bc2_sqrt), q.eps);
    p               = __fsub_rn(p, __fdiv_rn(__fmul_rn(q.step_size, m), den));
    q.p[i] = p;
    q.m[i] = m;
    q.v[i] = v;
}

__global__ void geometry_adam_kernel(
    const int64_t N,
    const GeomAdamParam means,  // [N, 3]
    const GeomAdamParam quats,  // [N, 4]
    const GeomAdamParam scales, // [N, 3] log-scales, act = exp
    const GeomAdamParam opac    // [N] logits, act = sigmoid
)
{
    // One thread per element of the concatenated [N*3 | N*4 | N*3 | N] range,
    // so every array is swept with coalesced accesses.
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < 3 * N)
    {
        geom_adam_elem(means, idx, ACT_NONE);
    }
    else if(idx < 7 * N)
    {
        geom_adam_elem(quats, idx - 3 * N, ACT_NONE);
    }
    else if(idx < 10 * N)
    {
        geom_adam_elem(scales, idx - 7 * N, ACT_EXP);
    }
    else if(idx < 11 * N)
    {
        geom_adam_elem(opac, idx - 10 * N, ACT_SIGMOID);
    }
}

static GeomAdamParam make_param(
    at::Tensor &p, at::Tensor &m, at::Tensor &v, const at::optional<at::Tensor> &g,
    const at::optional<at::Tensor> &act, double beta1, double beta2, double eps, double step_size, double bc2_sqrt
)
{
    CHECK_INPUT(p);
    CHECK_INPUT(m);
    CHECK_INPUT(v);
    TORCH_CHECK(m.sizes() == p.sizes() && v.sizes() == p.sizes(), "Adam state shape mismatch");
    if(g.has_value())
    {
        CHECK_INPUT(g.value());
        TORCH_CHECK(g.value().numel() == p.numel(), "gradient shape mismatch");
    }
    if(act.has_value())
    {
        CHECK_INPUT(act.value());
        TORCH_CHECK(act.value().numel() == p.numel(), "activation shape mismatch");
    }
    return GeomAdamParam{
        p.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        g.has_value() ? g.value().const_data_ptr<float>() : nullptr,
        act.has_value() ? act.value().const_data_ptr<float>() : nullptr,
        static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps),
        static_cast<float>(step_size), static_cast<float>(bc2_sqrt)};
}

// hp_* = [beta1, beta2, eps, step_size (lr / bias_correction1), bias_correction2_sqrt]
void geometry_adam(
    at::Tensor &means, at::Tensor &means_m, at::Tensor &means_v, const at::optional<at::Tensor> &g_means,
    at::ArrayRef<double> hp_means,
    at::Tensor &quats, at::Tensor &quats_m, at::Tensor &quats_v, const at::optional<at::Tensor> &g_quats,
    at::ArrayRef<double> hp_quats,
    at::Tensor &scales, at::Tensor &scales_m, at::Tensor &scales_v, const at::optional<at::Tensor> &g_scales,
    const at::Tensor &scales_act, at::ArrayRef<double> hp_scales,
    at::Tensor &opac, at::Tensor &opac_m, at::Tensor &opac_v, const at::optional<at::Tensor> &g_opac,
    const at::Tensor &opac_act, at::ArrayRef<double> hp_opac
)
{
    DEVICE_GUARD(means);
    const int64_t N = means.size(0);
    TORCH_CHECK(means.numel() == 3 * N && quats.numel() == 4 * N && scales.numel() == 3 * N && opac.numel() == N,
                "geometry_adam expects means [N,3], quats [N,4], scales [N,3], opacities [N]");
    for(auto hp: {hp_means, hp_quats, hp_scales, hp_opac})
    {
        TORCH_CHECK(hp.size() == 5, "hyper-parameter lists must have 5 entries");
    }
    auto pm = make_param(means, means_m, means_v, g_means, c10::nullopt, hp_means[0], hp_means[1], hp_means[2], hp_means[3], hp_means[4]);
    auto pq = make_param(quats, quats_m, quats_v, g_quats, c10::nullopt, hp_quats[0], hp_quats[1], hp_quats[2], hp_quats[3], hp_quats[4]);
    auto ps = make_param(scales, scales_m, scales_v, g_scales, scales_act, hp_scales[0], hp_scales[1], hp_scales[2], hp_scales[3], hp_scales[4]);
    auto po = make_param(opac, opac_m, opac_v, g_opac, opac_act, hp_opac[0], hp_opac[1], hp_opac[2], hp_opac[3], hp_opac[4]);
    const int64_t total = 11 * N;
    if(total == 0)
    {
        return;
    }
    constexpr unsigned threads = 256;
    geometry_adam_kernel<<<static_cast<unsigned>((total + threads - 1) / threads), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N, pm, pq, ps, po
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_IMPL(gsplat, CUDA, m)
{
    m.impl("geometry_adam", &geometry_adam);
}
} // namespace gsplat
