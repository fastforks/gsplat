/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

namespace at
{
class Tensor;
}

namespace gsplat
{
// Highest SH degree supported by the CUDA evaluation and backward kernels.
inline constexpr int SH_MAX_DEGREE = 4;

// Autograd-aware SH color evaluation; declared here for cross-TU orchestration
// callers (Rendering.cpp). Returns colors only.
at::Tensor spherical_harmonics(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs = c10::nullopt
);

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, D]
);

void launch_spherical_harmonics_l1_plus_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, D]
);

void launch_spherical_harmonics_fwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, D]
);

void launch_spherical_harmonics_l1_plus_fwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, D]
);

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, D]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats,
    at::optional<at::Tensor> v_viewmats_rs
);

void launch_spherical_harmonics_l1_plus_bwd_adam_kernel(
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    at::Tensor coeffs,
    const at::Tensor v_colors,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    at::optional<at::Tensor> v_means,
    const float step_size,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction2_sqrt,
    at::optional<at::Tensor> sh0,
    at::optional<at::Tensor> sh0_exp_avg,
    at::optional<at::Tensor> sh0_exp_avg_sq,
    const float sh0_step_size,
    const float sh0_beta1,
    const float sh0_beta2,
    const float sh0_eps,
    const float sh0_bias_correction2_sqrt
);

void launch_sh_colors_lazy_fwd_kernel(
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor mask,
    const at::Tensor sh0,
    const at::Tensor sh0_m,
    const at::Tensor sh0_v,
    const at::Tensor shN,
    const at::Tensor shN_m,
    const at::Tensor shN_v,
    const at::Tensor last,
    const int step,
    const at::Tensor tabs,
    const float sh0_beta1,
    const float sh0_beta2,
    const float sh0_eps,
    const float shN_beta1,
    const float shN_beta2,
    const float shN_eps,
    at::Tensor colors
);

void launch_sh_colors_lazy_bwd_adam_kernel(
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor v_colors,
    at::Tensor sh0,
    at::Tensor sh0_m,
    at::Tensor sh0_v,
    at::Tensor shN,
    at::Tensor shN_m,
    at::Tensor shN_v,
    at::Tensor last,
    const int step,
    const at::Tensor tabs,
    const float sh0_beta1,
    const float sh0_beta2,
    const float sh0_eps,
    const float shN_beta1,
    const float shN_beta2,
    const float shN_eps,
    at::optional<at::Tensor> v_means
);

void launch_sh_lazy_catchup_kernel(
    at::Tensor p,
    at::Tensor m,
    at::Tensor v,
    const at::Tensor last,
    const int step,
    const at::Tensor step_size_tab,
    const at::Tensor bc2_sqrt_tab,
    const float beta1,
    const float beta2,
    const float eps
);

void launch_sh_colors_fused_fwd_kernel(
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor sh0,
    const at::Tensor coeffs,
    at::Tensor colors
);

void launch_spherical_harmonics_l1_plus_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, D]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats,
    at::optional<at::Tensor> v_viewmats_rs
);

// Precompute R^T t, averaged across shutter endpoints when viewmats_rs is
// present, once per image. The result is flattened as [num_viewmats, 3].
at::Tensor precompute_spherical_harmonics_camera_offsets(
    const at::Tensor viewmats, const at::optional<at::Tensor> viewmats_rs
);

// Reduce intermediate view-direction gradients into world-to-camera matrix
// gradients. The regular CUDA path assigns the result; the multi-GPU path
// atomically accumulates one Gaussian chunk.
enum class ViewmatGradientUpdate
{
    Assign,
    SystemAtomicAdd
};

template<ViewmatGradientUpdate Update>
void launch_spherical_harmonics_view_direction_vjp_reduction(
    const int64_t N,
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const at::Tensor viewmats,
    const at::Tensor v_viewdirs,
    at::optional<at::Tensor> v_viewmats,
    const at::optional<at::Tensor> viewmats_rs = c10::nullopt,
    const at::optional<at::Tensor> batch_ids   = c10::nullopt,
    const at::optional<at::Tensor> camera_ids  = c10::nullopt,
    at::optional<at::Tensor> v_viewmats_rs     = c10::nullopt
);

// Fused forward assembly of proj_features = [SH colors | extra | (depth)] for
// the unpacked rasterization path. Writes each complete output row in one
// coalesced pass, replacing the SH-eval + cat(color, extra) + cat(.., depth)
// chain.
void launch_assemble_proj_features_unpacked_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t degrees_to_use,
    const uint32_t Dc,
    const uint32_t E,
    const uint32_t color_post,
    const uint32_t extra_post,
    const bool has_depth,
    const bool depth_is_zero,
    const bool extra_has_c,
    const at::Tensor means,                     // [B, N, 3]
    const at::Tensor viewmats,                  // [B, C, 4, 4]
    const at::optional<at::Tensor> viewmats_rs, // [B, C, 4, 4]
    const at::Tensor coeffs,                    // [N, K, Dc]
    const at::optional<at::Tensor> extra,       // [B, C, N, E] or [B, N, E]
    const at::optional<at::Tensor> depths,      // [B, C, N]
    const at::optional<at::Tensor> masks,       // [B, C, N]
    at::Tensor out,                             // [B, C, N, Dc + E + has_depth]
    const at::optional<at::Tensor> relu_mask    // [B, C, N, Dc]
);

// Autograd-aware fused assembly of proj_features = [SH colors | extra | (depth)]
// for the unpacked rasterization path. Forward writes the whole tensor in one
// coalesced kernel (folding SH eval + the +0.5/relu post-op + extra read + depth
// write); backward routes the color-slice gradient through the SH backward and
// passes extra/depth gradients straight through. Declared for cross-TU callers
// (Rendering.cpp orchestration). Returns proj_features [*batch, C, N, width].
at::Tensor assemble_proj_features(
    int64_t degrees_to_use,
    int64_t B,
    int64_t C,
    int64_t N,
    int64_t Dc,
    int64_t E,
    int64_t color_post,
    int64_t extra_post,
    bool has_depth,
    bool depth_is_zero,
    bool extra_has_c,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::optional<at::Tensor> &viewmats_rs,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &extra,
    const at::optional<at::Tensor> &depths,
    const at::optional<at::Tensor> &masks
);

// Forward-only dispatcher op: assembles proj_features in place into `out`.
void assemble_proj_features_unpacked_fwd(
    int64_t degrees_to_use,
    int64_t B,
    int64_t C,
    int64_t N,
    int64_t Dc,
    int64_t E,
    int64_t color_post,
    int64_t extra_post,
    bool has_depth,
    bool depth_is_zero,
    bool extra_has_c,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::optional<at::Tensor> &viewmats_rs,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &extra,
    const at::optional<at::Tensor> &depths,
    const at::optional<at::Tensor> &masks,
    at::Tensor &out,
    const at::optional<at::Tensor> &relu_mask
);

void launch_spherical_harmonics_bwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, D]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats
);

void launch_spherical_harmonics_l1_plus_bwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, D]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats
);
} // namespace gsplat
