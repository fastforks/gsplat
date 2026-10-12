/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Fused per-step densification statistics for DefaultStrategy (dense layout).

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda/cmath>
#include <torch/library.h>

#include "Common.h"

namespace gsplat
{

// For each gaussian n, over the C images where it is visible (both radii > 0):
//   grad2d[n] += sum_c |(g[c,n,0] * sx, g[c,n,1] * sy)|_2
//   count[n]  += number of such images
// matching DefaultStrategy._update_state's tensor ops. Correctly rounded
// intrinsics keep the arithmetic identical to torch's (no --use_fast_math
// contraction / approximate sqrt).
__global__ void strategy_update_state_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ grads,  // [C, N, 2], rows gs apart
    const int64_t gs,
    const int32_t *__restrict__ radii, // [C, N, 2]
    const float sx,
    const float sy,
    float *__restrict__ grad2d, // [N]
    float *__restrict__ count   // [N]
)
{
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N)
    {
        return;
    }
    float g_sum = 0.f, c_sum = 0.f;
    for(uint32_t c = 0; c < C; ++c)
    {
        const int64_t i = static_cast<int64_t>(c) * N + n;
        const int2 r    = reinterpret_cast<const int2 *>(radii)[i];
        if(!(r.x > 0 && r.y > 0))
        {
            continue;
        }
        const float2 g = gs == 2 ? reinterpret_cast<const float2 *>(grads)[i] : make_float2(grads[i * gs], grads[i * gs + 1]);
        const float gx = __fmul_rn(g.x, sx), gy = __fmul_rn(g.y, sy);
        g_sum          = __fadd_rn(g_sum, __fsqrt_rn(__fadd_rn(__fmul_rn(gx, gx), __fmul_rn(gy, gy))));
        c_sum          = __fadd_rn(c_sum, 1.f);
    }
    grad2d[n] = __fadd_rn(grad2d[n], g_sum);
    count[n]  = __fadd_rn(count[n], c_sum);
}

void strategy_update_state(
    const at::Tensor &grads,
    const at::Tensor &radii,
    double sx,
    double sy,
    at::Tensor &grad2d,
    at::Tensor &count
)
{
    DEVICE_GUARD(grads);
    // grads may be a row-strided view (the rasterizer's packed gradient buffer).
    CHECK_DEVICE(grads);
    TORCH_CHECK(grads.dim() == 3 && grads.stride(-1) == 1 && grads.stride(0) == grads.size(1) * grads.stride(1),
                "grads must be [C, N, 2] with evenly strided rows");
    CHECK_INPUT(radii);
    CHECK_INPUT(grad2d);
    CHECK_INPUT(count);
    TORCH_CHECK(grads.size(-1) == 2, "grads must be [C, N, 2]");
    TORCH_CHECK(radii.sizes() == grads.sizes() && radii.scalar_type() == at::kInt, "radii must be int32 [C, N, 2]");
    TORCH_CHECK(grads.scalar_type() == at::kFloat, "grads must be float32");
    const uint32_t C = grads.size(0), N = grads.size(1);
    TORCH_CHECK(grad2d.numel() == N && count.numel() == N, "state must be [N]");
    if(N == 0)
    {
        return;
    }
    constexpr unsigned threads = 256;
    strategy_update_state_kernel<<<(N + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        C,
        N,
        grads.const_data_ptr<float>(),
        grads.stride(1),
        radii.const_data_ptr<int32_t>(),
        static_cast<float>(sx),
        static_cast<float>(sy),
        grad2d.data_ptr<float>(),
        count.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_IMPL(gsplat, CUDA, m)
{
    m.impl("strategy_update_state", &strategy_update_state);
}
} // namespace gsplat
