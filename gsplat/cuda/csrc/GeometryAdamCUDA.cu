/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Adam for the geometry parameters (means, quats, log-scales, opacity logits)
// in one pass, with the scale (exp) and opacity (sigmoid) activation backward
// fused in. Each (parameter, element) is updated exactly like
// torch.optim.Adam (weight_decay=0, amsgrad=False) given its gradient.

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "Common.h"
#include "Utils.cuh"
#include "AccuTile.cuh"

namespace gsplat
{

struct GeomAdamParam
{
    float *p, *m, *v;
    const float *g;  // gradient w.r.t. the activated value (nullptr: zero)
    const float *g2; // optional second gradient contribution, added to g
    const float *act; // activation output for the chain rule (nullptr: identity)
    bool act_from_p;  // recompute the activation from p instead of reading act
    float beta1, beta2, eps, step_size, bc2_sqrt;
    bool skip; // no gradient this step: torch.optim.Adam skips the parameter
};

enum GeomAct
{
    ACT_NONE = 0,
    ACT_EXP  = 1,
    ACT_SIGMOID = 2
};

extern "C" __device__ float __nv_expf(float);

// torch.exp / torch.sigmoid on CUDA, bit for bit. The library expf is called
// directly because -use_fast_math would turn expf() into __expf(). Results in
// the denormal range (|x| > ~87) are flushed to zero under fast math, where
// torch keeps the denormal.
__device__ __forceinline__ float act_exp(const float x) { return __nv_expf(x); }
__device__ __forceinline__ float act_sigmoid(const float x)
{
    return __fdiv_rn(1.f, __fadd_rn(1.f, __nv_expf(-x)));
}

// The torch.optim.Adam update of element i given its (activated) gradient and
// current value p; writes p, m, v.
__device__ __forceinline__ void geom_adam_apply(const GeomAdamParam &q, const int64_t i, const float grad, float p)
{
    float m = q.m[i], v = q.v[i];
    m               = __fmaf_rn(q.beta1, m, __fmul_rn(__fsub_rn(1.f, q.beta1), grad));
    v               = __fmaf_rn(q.beta2, v, __fmul_rn(__fmul_rn(__fsub_rn(1.f, q.beta2), grad), grad));
    const float den = __fadd_rn(__fdiv_rn(__fsqrt_rn(v), q.bc2_sqrt), q.eps);
    p               = __fsub_rn(p, __fdiv_rn(__fmul_rn(q.step_size, m), den));
    q.p[i] = p;
    q.m[i] = m;
    q.v[i] = v;
}

__device__ __forceinline__ void geom_adam_elem(const GeomAdamParam &q, const int64_t i, const int act)
{
    if(q.skip)
    {
        return;
    }
    float grad = 0.f;
    const float p0 = q.p[i];
    if(q.g != nullptr || q.g2 != nullptr)
    {
        if(q.g != nullptr && q.g2 != nullptr)
            grad = __fadd_rn(q.g[i], q.g2[i]); // what autograd's accumulation would do
        else
            grad = q.g != nullptr ? q.g[i] : q.g2[i];
        if(act == ACT_EXP)
        {
            grad = __fmul_rn(grad, q.act_from_p ? act_exp(p0) : q.act[i]); // exp backward: grad * result
        }
        else if(act == ACT_SIGMOID)
        {
            const float y = q.act_from_p ? act_sigmoid(p0) : q.act[i]; // sigmoid backward: grad * (1 - y) * y
            grad          = __fmul_rn(__fmul_rn(grad, __fsub_rn(1.f, y)), y);
        }
    }
    geom_adam_apply(q, i, grad, p0);
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
    at::Tensor &p, at::Tensor &m, at::Tensor &v, const at::optional<at::Tensor> &g, const at::optional<at::Tensor> &g2,
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
    if(g2.has_value())
    {
        CHECK_INPUT(g2.value());
        TORCH_CHECK(g2.value().numel() == p.numel(), "gradient shape mismatch");
    }
    if(act.has_value())
    {
        CHECK_INPUT(act.value());
        TORCH_CHECK(act.value().numel() == p.numel(), "activation shape mismatch");
    }
    return GeomAdamParam{
        p.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        g.has_value() ? g.value().const_data_ptr<float>() : nullptr,
        g2.has_value() ? g2.value().const_data_ptr<float>() : nullptr,
        act.has_value() ? act.value().const_data_ptr<float>() : nullptr,
        false, static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps),
        static_cast<float>(step_size), static_cast<float>(bc2_sqrt), !g.has_value() && !g2.has_value()};
}

// hp_* = [beta1, beta2, eps, step_size (lr / bias_correction1), bias_correction2_sqrt]
void geometry_adam(
    at::Tensor &means, at::Tensor &means_m, at::Tensor &means_v, const at::optional<at::Tensor> &g_means,
    const at::optional<at::Tensor> &g2_means, at::ArrayRef<double> hp_means,
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
    auto pm = make_param(means, means_m, means_v, g_means, g2_means, c10::nullopt, hp_means[0], hp_means[1], hp_means[2], hp_means[3], hp_means[4]);
    auto pq = make_param(quats, quats_m, quats_v, g_quats, c10::nullopt, c10::nullopt, hp_quats[0], hp_quats[1], hp_quats[2], hp_quats[3], hp_quats[4]);
    auto ps = make_param(scales, scales_m, scales_v, g_scales, c10::nullopt, scales_act, hp_scales[0], hp_scales[1], hp_scales[2], hp_scales[3], hp_scales[4]);
    auto po = make_param(opac, opac_m, opac_v, g_opac, c10::nullopt, opac_act, hp_opac[0], hp_opac[1], hp_opac[2], hp_opac[3], hp_opac[4]);
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

// ---------------------------------------------------------------------------
// Projection backward (EWA, pinhole, one camera, no compensation) fused with
// the geometry Adam step. One thread per gaussian computes the gradients of
// its mean / quaternion / scale exactly as projection_ewa_3dgs_fused_bwd_kernel
// does (same helpers; with one camera that kernel's warp groups have size 1
// and its atomics add onto zero) and applies the Adam updates right away, so
// the gradients never go through memory and need no zero-fill. Culled
// gaussians get zero gradients, as in the unfused path.
// ---------------------------------------------------------------------------
__global__ void projection_geometry_adam_kernel(
    const int64_t N,
    const float *__restrict__ viewmat, // [4, 4]
    const float *__restrict__ K,       // [3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const int32_t *__restrict__ radii,     // [N, 2]
    const float *__restrict__ conics,      // [N, 3]
    const float *__restrict__ v_means2d,   // [N, 2], rows sm apart
    const float *__restrict__ v_depths,    // [N] or nullptr (zero)
    const float *__restrict__ v_conics,    // [N, 3], rows sc apart
    const int64_t sm,
    const int64_t sc,
    const GeomAdamParam means,             // g2 = extra means gradient (optional), g unused
    const GeomAdamParam quats,
    const GeomAdamParam scales,            // log-scales; exp recomputed (projection input)
    const GeomAdamParam opac,              // g unused; sigmoid recomputed
    const float *__restrict__ g_opac,      // [N] opacity gradient, rows so apart (nullptr: zero)
    const int64_t so,
    const bool g_opac_visible_only         // g_opac is zero for culled gaussians (not read there)
)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= N)
    {
        return;
    }
    const vec3 mean  = vec3(means.p[i * 3], means.p[i * 3 + 1], means.p[i * 3 + 2]);
    const vec4 quat  = vec4(quats.p[i * 4], quats.p[i * 4 + 1], quats.p[i * 4 + 2], quats.p[i * 4 + 3]);
    // scales holds log-scales; the activation is recomputed (as in the forward).
    const float ls[3] = {scales.p[i * 3], scales.p[i * 3 + 1], scales.p[i * 3 + 2]};
    const vec3 scale  = vec3(act_exp(ls[0]), act_exp(ls[1]), act_exp(ls[2]));

    vec3 v_mean(0.f);
    vec4 v_quat(0.f);
    vec3 v_scale(0.f);
    const bool visible = radii[i * 2] > 0 && radii[i * 2 + 1] > 0;
    if(visible)
    {
        const mat2 covar2d_inv   = mat2(conics[i * 3], conics[i * 3 + 1], conics[i * 3 + 1], conics[i * 3 + 2]);
        const float *vc          = v_conics + i * sc;
        const mat2 v_covar2d_inv = mat2(vc[0], vc[1] * .5f, vc[1] * .5f, vc[2]);
        mat2 v_covar2d(0.f);
        inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

        const mat3 R = mat3(
            viewmat[0], viewmat[4], viewmat[8], // 1st column
            viewmat[1], viewmat[5], viewmat[9], // 2nd column
            viewmat[2], viewmat[6], viewmat[10] // 3rd column
        );
        const vec3 t = vec3(viewmat[3], viewmat[7], viewmat[11]);
        mat3 covar;
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
        vec3 mean_c;
        posW2C(R, t, mean, mean_c);
        mat3 covar_c;
        covarW2C(R, covar, covar_c);

        mat3 v_covar_c(0.f);
        vec3 v_mean_c(0.f);
        persp_proj_vjp(
            mean_c, covar_c, K[0], K[4], K[2], K[5], image_width, image_height, v_covar2d,
            vec2(v_means2d[i * sm], v_means2d[i * sm + 1]), v_mean_c, v_covar_c
        );
        v_mean_c.z += v_depths != nullptr ? v_depths[i] : 0.f;

        mat3 v_covar(0.f);
        mat3 v_R(0.f);
        vec3 v_t(0.f);
        posW2C_VJP(R, t, mean, v_mean_c, v_R, v_t, v_mean);
        covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);
        const mat3 rotmat = quat_to_rotmat(quat);
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
    }

    // Same per-element arithmetic as geom_adam_elem (gradient + optional extra,
    // activation backward), applied to the values read above.
#pragma unroll
    for(int k = 0; k < 3; ++k)
    {
        const int64_t e = i * 3 + k;
        const float g   = means.g2 != nullptr ? __fadd_rn(v_mean[k], means.g2[e]) : v_mean[k];
        geom_adam_apply(means, e, g, mean[k]);
    }
#pragma unroll
    for(int k = 0; k < 4; ++k)
    {
        geom_adam_apply(quats, i * 4 + k, v_quat[k], quat[k]);
    }
#pragma unroll
    for(int k = 0; k < 3; ++k)
    {
        const int64_t e = i * 3 + k;
        geom_adam_apply(scales, e, __fmul_rn(v_scale[k], scale[k]), ls[k]);
    }
    // As geom_adam_elem(opac, i, ACT_SIGMOID) with g = g_opac (strided).
    if(!opac.skip)
    {
        const float p0 = opac.p[i];
        float grad     = 0.f;
        if(g_opac != nullptr)
        {
            grad          = (visible || !g_opac_visible_only) ? g_opac[i * so] : 0.f;
            const float y = act_sigmoid(p0);
            grad          = __fmul_rn(__fmul_rn(grad, __fsub_rn(1.f, y)), y);
        }
        geom_adam_apply(opac, i, grad, p0);
    }
}

// Gradient-free parts of the argument lists are as in geometry_adam; the
// projection inputs/outputs are those of projection_ewa_3dgs_fused (B = C = 1).
void projection_geometry_adam(
    at::Tensor &means, at::Tensor &means_m, at::Tensor &means_v, const at::optional<at::Tensor> &g2_means,
    at::ArrayRef<double> hp_means,
    at::Tensor &quats, at::Tensor &quats_m, at::Tensor &quats_v, at::ArrayRef<double> hp_quats,
    at::Tensor &scales, at::Tensor &scales_m, at::Tensor &scales_v, at::ArrayRef<double> hp_scales,
    at::Tensor &opac, at::Tensor &opac_m, at::Tensor &opac_v, const at::optional<at::Tensor> &g_opac,
    at::ArrayRef<double> hp_opac,
    const at::Tensor &viewmats, const at::Tensor &Ks, int64_t image_width, int64_t image_height,
    const at::Tensor &radii, const at::Tensor &conics, const at::Tensor &v_means2d,
    const at::optional<at::Tensor> &v_depths, const at::Tensor &v_conics, bool g_opac_visible_only
)
{
    DEVICE_GUARD(means);
    const int64_t N = means.size(0);
    TORCH_CHECK(means.numel() == 3 * N && quats.numel() == 4 * N && scales.numel() == 3 * N && opac.numel() == N,
                "projection_geometry_adam expects means [N,3], quats [N,4], scales [N,3], opacities [N]");
    TORCH_CHECK(viewmats.numel() == 16 && Ks.numel() == 9, "projection_geometry_adam supports one camera");
    for(auto t: {viewmats, Ks, radii, conics})
    {
        CHECK_INPUT(t);
    }
    // Gradients may be row-strided views (e.g. of the rasterizer's packed
    // gradient buffer): [1, N, k] with unit inner stride.
    auto row_stride = [N](const at::Tensor &t, int64_t k) -> int64_t
    {
        CHECK_DEVICE(t);
        TORCH_CHECK(t.scalar_type() == at::kFloat && t.numel() == k * N && t.size(-1) == (k == 1 ? N : k),
                    "projection_geometry_adam: bad gradient shape");
        TORCH_CHECK(k == 1 || t.stride(-1) == 1, "projection_geometry_adam: gradient needs unit inner stride");
        return k == 1 ? t.stride(-1) : t.stride(-2);
    };
    const int64_t sm = row_stride(v_means2d, 2), sc = row_stride(v_conics, 3);
    const int64_t so = g_opac.has_value() ? row_stride(g_opac.value(), 1) : 1;
    TORCH_CHECK(viewmats.scalar_type() == at::kFloat && Ks.scalar_type() == at::kFloat, "float32 cameras expected");
    TORCH_CHECK(radii.numel() == 2 * N && conics.numel() == 3 * N, "projection tensors must be [1, N, ...]");
    if(v_depths.has_value())
    {
        CHECK_INPUT(v_depths.value());
        TORCH_CHECK(v_depths.value().numel() == N, "v_depths must be [1, N]");
    }
    for(auto hp: {hp_means, hp_quats, hp_scales, hp_opac})
    {
        TORCH_CHECK(hp.size() == 5, "hyper-parameter lists must have 5 entries");
    }
    const at::optional<at::Tensor> none = c10::nullopt;
    // means / quats / scales always have a gradient here (the projection's);
    // a dummy g marks them as such (it is not read).
    auto pm = make_param(means, means_m, means_v, means, g2_means, none, hp_means[0], hp_means[1], hp_means[2], hp_means[3], hp_means[4]);
    auto pq = make_param(quats, quats_m, quats_v, quats, none, none, hp_quats[0], hp_quats[1], hp_quats[2], hp_quats[3], hp_quats[4]);
    auto ps = make_param(scales, scales_m, scales_v, scales, none, none, hp_scales[0], hp_scales[1], hp_scales[2], hp_scales[3], hp_scales[4]);
    auto po = make_param(opac, opac_m, opac_v, none, none, none, hp_opac[0], hp_opac[1], hp_opac[2], hp_opac[3], hp_opac[4]);
    po.skip       = !g_opac.has_value();
    ps.act_from_p = true;
    po.act_from_p = true;
    if(N == 0)
    {
        return;
    }
    constexpr unsigned threads = 256;
    projection_geometry_adam_kernel<<<static_cast<unsigned>((N + threads - 1) / threads), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N, viewmats.const_data_ptr<float>(), Ks.const_data_ptr<float>(), static_cast<uint32_t>(image_width),
        static_cast<uint32_t>(image_height), radii.const_data_ptr<int32_t>(), conics.const_data_ptr<float>(),
        v_means2d.const_data_ptr<float>(), v_depths.has_value() ? v_depths.value().const_data_ptr<float>() : nullptr,
        v_conics.const_data_ptr<float>(), sm, sc, pm, pq, ps, po,
        g_opac.has_value() ? g_opac.value().const_data_ptr<float>() : nullptr, so, g_opac_visible_only
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Forward counterpart of projection_geometry_adam: projection_ewa_3dgs_fused
// forward (pinhole, one camera, no compensation, radius_clip 0) that reads the
// raw log-scales / opacity logits and applies exp / sigmoid on load, so no
// activated copies are materialized. Writes sigmoid(logit) for every gaussian
// (the rasterizer's opacities). Same arithmetic as the upstream kernel.
__global__ void projection_act_fwd_kernel(
    const int64_t N,
    const float *__restrict__ means,      // [N, 3]
    const float *__restrict__ quats,      // [N, 4]
    const float *__restrict__ log_scales, // [N, 3]
    const float *__restrict__ logits,     // [N]
    const float *__restrict__ viewmat,    // [4, 4]
    const float *__restrict__ K,          // [3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    int32_t *__restrict__ radii,  // [N, 2]
    float *__restrict__ means2d,  // [N, 2]
    float *__restrict__ depths,   // [N]
    float *__restrict__ conics,   // [N, 3]
    float *__restrict__ opac_out, // [N]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    int32_t *__restrict__ tiles_per_gauss // [N] AccuTile count (intersect_tile's first pass)
)
{
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(gid >= N)
    {
        return;
    }
    const float opacity = act_sigmoid(logits[gid]);
    opac_out[gid]       = opacity;
    tiles_per_gauss[gid] = 0; // overwritten below for visible gaussians

    mat3 R = mat3(
        viewmat[0], viewmat[4], viewmat[8], // 1st column
        viewmat[1], viewmat[5], viewmat[9], // 2nd column
        viewmat[2], viewmat[6], viewmat[10] // 3rd column
    );
    vec3 t = vec3(viewmat[3], viewmat[7], viewmat[11]);
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means + gid * 3), mean_c);
    if(mean_c.z < near_plane || mean_c.z > far_plane)
    {
        radii[gid * 2]     = 0;
        radii[gid * 2 + 1] = 0;
        return;
    }
    const float *ls  = log_scales + gid * 3;
    const vec3 scale = vec3(act_exp(ls[0]), act_exp(ls[1]), act_exp(ls[2]));
    mat3 covar;
    quat_scale_to_covar_preci(glm::make_vec4(quats + gid * 4), scale, &covar, nullptr);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);
    mat2 covar2d;
    vec2 mean2d;
    persp_proj(mean_c, covar_c, K[0], K[4], K[2], K[5], image_width, image_height, covar2d, mean2d);

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if(det <= 0.f)
    {
        radii[gid * 2]     = 0;
        radii[gid * 2 + 1] = 0;
        return;
    }
    mat2 covar2d_inv = glm::inverse(covar2d);

    if(opacity < ALPHA_THRESHOLD)
    {
        radii[gid * 2]     = 0;
        radii[gid * 2 + 1] = 0;
        return;
    }
    const float extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    float radius_x     = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y     = ceilf(extend * sqrtf(covar2d[1][1]));
    if(radius_x <= 0.f && radius_y <= 0.f)
    {
        radii[gid * 2]     = 0;
        radii[gid * 2 + 1] = 0;
        return;
    }
    if(mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width || mean2d.y + radius_y <= 0
       || mean2d.y - radius_y >= image_height)
    {
        radii[gid * 2]     = 0;
        radii[gid * 2 + 1] = 0;
        return;
    }
    radii[gid * 2]       = (int32_t)radius_x;
    radii[gid * 2 + 1]   = (int32_t)radius_y;
    means2d[gid * 2]     = mean2d.x;
    means2d[gid * 2 + 1] = mean2d.y;
    depths[gid]          = mean_c.z;
    conics[gid * 3]      = covar2d_inv[0][0];
    conics[gid * 3 + 1]  = covar2d_inv[0][1];
    conics[gid * 3 + 2]  = covar2d_inv[1][1];

    // Tile count exactly as intersect_tile_kernel's first pass (dense, one
    // image, AccuTile path) computes it from the values just written.
    if((int32_t)radius_x <= 0 || (int32_t)radius_y <= 0)
    {
        return;
    }
    const float A = covar2d_inv[0][0], B = covar2d_inv[0][1], C = covar2d_inv[1][1];
    const float2 m2d       = {mean2d.x, mean2d.y};
    const AccuTileSetup su = accutile_setup(A, B, C, opacity, m2d, tile_size, tile_width, tile_height);
    if(su.empty)
    {
        return;
    }
    int64_t unused = 0;
    tiles_per_gauss[gid] = static_cast<int32_t>(accutile_process_tiles(
        A, B, C, su.disc, su.t, m2d, su.bbox_min, su.bbox_max, su.bbox_argmin, su.bbox_argmax, su.rect_min,
        su.rect_max, tile_size, tile_width, tile_width * tile_height, su.isY, 0, 0,
        static_cast<int64_t>(tile_width) * tile_height, 0, 0, 0, 0, nullptr, nullptr, &unused
    ));
}

// Returns radii [1, N, 2], means2d [1, N, 2], depths [1, N], conics [1, N, 3],
// opacities [N] (sigmoid of the logits).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_act_fwd(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &log_scales, const at::Tensor &logits,
    const at::Tensor &viewmats, const at::Tensor &Ks, int64_t image_width, int64_t image_height, double eps2d,
    double near_plane, double far_plane, int64_t tile_size, int64_t tile_width, int64_t tile_height
)
{
    DEVICE_GUARD(means);
    for(auto t: {means, quats, log_scales, logits, viewmats, Ks})
    {
        CHECK_INPUT(t);
        TORCH_CHECK(t.scalar_type() == at::kFloat, "projection_act_fwd expects float32");
    }
    const int64_t N = means.size(0);
    TORCH_CHECK(means.numel() == 3 * N && quats.numel() == 4 * N && log_scales.numel() == 3 * N && logits.numel() == N,
                "projection_act_fwd expects means [N,3], quats [N,4], scales [N,3], opacities [N]");
    TORCH_CHECK(viewmats.numel() == 16 && Ks.numel() == 9, "projection_act_fwd supports one camera");
    auto opt          = means.options();
    at::Tensor radii  = at::empty({1, N, 2}, opt.dtype(at::kInt));
    at::Tensor m2d    = at::empty({1, N, 2}, opt);
    at::Tensor depths = at::empty({1, N}, opt);
    at::Tensor conics = at::empty({1, N, 3}, opt);
    at::Tensor opac   = at::empty({N}, opt);
    at::Tensor tiles  = at::empty({1, N}, opt.dtype(at::kInt));
    if(N > 0)
    {
        constexpr unsigned threads = 256;
        projection_act_fwd_kernel<<<static_cast<unsigned>((N + threads - 1) / threads), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, means.const_data_ptr<float>(), quats.const_data_ptr<float>(), log_scales.const_data_ptr<float>(),
            logits.const_data_ptr<float>(), viewmats.const_data_ptr<float>(), Ks.const_data_ptr<float>(),
            static_cast<uint32_t>(image_width), static_cast<uint32_t>(image_height), static_cast<float>(eps2d),
            static_cast<float>(near_plane), static_cast<float>(far_plane), radii.data_ptr<int32_t>(),
            m2d.data_ptr<float>(), depths.data_ptr<float>(), conics.data_ptr<float>(), opac.data_ptr<float>(),
            static_cast<uint32_t>(tile_size), static_cast<uint32_t>(tile_width), static_cast<uint32_t>(tile_height),
            tiles.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {radii, m2d, depths, conics, opac, tiles};
}

TORCH_LIBRARY_IMPL(gsplat, CUDA, m)
{
    m.impl("geometry_adam", &geometry_adam);
    m.impl("projection_geometry_adam", &projection_geometry_adam);
    m.impl("projection_act_fwd", &projection_act_fwd);
}
} // namespace gsplat
