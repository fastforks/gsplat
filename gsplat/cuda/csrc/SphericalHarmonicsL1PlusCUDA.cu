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

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <array>
#include <cooperative_groups.h>
#include <vector>

#include "Common.h"
#include "Dispatch.h"
#include "SphericalHarmonics.h"
#include "SphericalHarmonics.cuh"
#include "Utils.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

template<typename scalar_t>
__device__ void sh_l1_plus_coeffs_to_color_fast(
    const uint32_t degree,  // degree of SH to be evaluated
    const uint32_t D,       // channels (coeffs last dim)
    const uint32_t c,       // output channel
    const vec3 &dir,        // [3]
    const scalar_t *coeffs, // [K - 1, D]
    // output
    at::opmath_type<scalar_t> *colors // [D]
)
{
    using opmath_t  = at::opmath_type<scalar_t>;
    opmath_t result = 0;
    if(degree >= 1)
    {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x     = dir.x * inorm;
        float y     = dir.y * inorm;
        float z     = dir.z * inorm;

        result += 0.48860251190292f
                * (-y * static_cast<opmath_t>(coeffs[c])
                   + z * static_cast<opmath_t>(coeffs[1 * D + c])
                   - x * static_cast<opmath_t>(coeffs[2 * D + c]));
        if(degree >= 2)
        {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1    = x * x - y * y;
            float fS1    = 2.f * x * y;
            float pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7   = fTmp0B * x;
            float pSH5   = fTmp0B * y;
            float pSH8   = 0.5462742152960395f * fC1;
            float pSH4   = 0.5462742152960395f * fS1;

            result += pSH4 * static_cast<opmath_t>(coeffs[3 * D + c])
                    + pSH5 * static_cast<opmath_t>(coeffs[4 * D + c])
                    + pSH6 * static_cast<opmath_t>(coeffs[5 * D + c])
                    + pSH7 * static_cast<opmath_t>(coeffs[6 * D + c])
                    + pSH8 * static_cast<opmath_t>(coeffs[7 * D + c]);
            if(degree >= 3)
            {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2    = x * fC1 - y * fS1;
                float fS2    = x * fS1 + y * fC1;
                float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13  = fTmp0C * x;
                float pSH11  = fTmp0C * y;
                float pSH14  = fTmp1B * fC1;
                float pSH10  = fTmp1B * fS1;
                float pSH15  = -0.5900435899266435f * fC2;
                float pSH9   = -0.5900435899266435f * fS2;

                result += pSH9 * static_cast<opmath_t>(coeffs[8 * D + c])
                        + pSH10 * static_cast<opmath_t>(coeffs[9 * D + c])
                        + pSH11 * static_cast<opmath_t>(coeffs[10 * D + c])
                        + pSH12 * static_cast<opmath_t>(coeffs[11 * D + c])
                        + pSH13 * static_cast<opmath_t>(coeffs[12 * D + c])
                        + pSH14 * static_cast<opmath_t>(coeffs[13 * D + c])
                        + pSH15 * static_cast<opmath_t>(coeffs[14 * D + c]);

                if(degree >= 4)
                {
                    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3    = x * fC2 - y * fS2;
                    float fS3    = x * fS2 + y * fC2;
                    float pSH20  = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
                    float pSH21  = fTmp0D * x;
                    float pSH19  = fTmp0D * y;
                    float pSH22  = fTmp1C * fC1;
                    float pSH18  = fTmp1C * fS1;
                    float pSH23  = fTmp2B * fC2;
                    float pSH17  = fTmp2B * fS2;
                    float pSH24  = 0.6258357354491763f * fC3;
                    float pSH16  = 0.6258357354491763f * fS3;

                    result += pSH16 * static_cast<opmath_t>(coeffs[15 * D + c])
                            + pSH17 * static_cast<opmath_t>(coeffs[16 * D + c])
                            + pSH18 * static_cast<opmath_t>(coeffs[17 * D + c])
                            + pSH19 * static_cast<opmath_t>(coeffs[18 * D + c])
                            + pSH20 * static_cast<opmath_t>(coeffs[19 * D + c])
                            + pSH21 * static_cast<opmath_t>(coeffs[20 * D + c])
                            + pSH22 * static_cast<opmath_t>(coeffs[21 * D + c])
                            + pSH23 * static_cast<opmath_t>(coeffs[22 * D + c])
                            + pSH24 * static_cast<opmath_t>(coeffs[23 * D + c]);
                }
            }
        }
    }

    colors[c] = result;
}

template<typename scalar_t>
__device__ void sh_l1_plus_coeffs_to_color_fast_vjp(
    const uint32_t degree,                     // degree of SH to be evaluated
    const uint32_t D,                          // channels (coeffs last dim)
    const uint32_t c,                          // output channel
    const vec3 &dir,                           // [3]
    const scalar_t *coeffs,                    // [K - 1, D]
    const at::opmath_type<scalar_t> *v_colors, // [D]
    // output
    // acc is a per-thread fp32 accumulator indexed by coefficient k (channel c is
    // fixed for this thread). The caller sums contributions from every batch here
    // in registers and writes the result to global memory once (casting to the
    // coeff dtype), so no global atomics or fp32 scratch buffer are needed.
    at::opmath_type<scalar_t> *acc, // [MAX_K]
    vec3 *v_dir                     // [3] optional
)
{
    using opmath_t          = at::opmath_type<scalar_t>;
    opmath_t v_colors_local = v_colors[c];

    if(degree < 1)
    {
        return;
    }
    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x     = dir.x * inorm;
    float y     = dir.y * inorm;
    float z     = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    acc[0] += -0.48860251190292f * y * v_colors_local;
    acc[1] += 0.48860251190292f * z * v_colors_local;
    acc[2] += -0.48860251190292f * x * v_colors_local;

    if(v_dir != nullptr)
    {
        v_x += -0.48860251190292f * static_cast<opmath_t>(coeffs[2 * D + c]) * v_colors_local;
        v_y += -0.48860251190292f * static_cast<opmath_t>(coeffs[c]) * v_colors_local;
        v_z += 0.48860251190292f * static_cast<opmath_t>(coeffs[1 * D + c]) * v_colors_local;
    }
    if(degree < 2)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float z2      = z * z;
    float fTmp0B  = -1.092548430592079f * z;
    float fC1     = x * x - y * y;
    float fS1     = 2.f * x * y;
    float pSH6    = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7    = fTmp0B * x;
    float pSH5    = fTmp0B * y;
    float pSH8    = 0.5462742152960395f * fC1;
    float pSH4    = 0.5462742152960395f * fS1;
    acc[3]       += pSH4 * v_colors_local;
    acc[4]       += pSH5 * v_colors_local;
    acc[5]       += pSH6 * v_colors_local;
    acc[6]       += pSH7 * v_colors_local;
    acc[7]       += pSH8 * v_colors_local;

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y, pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if(v_dir != nullptr)
    {
        fTmp0B_z = -1.092548430592079f;
        fC1_x    = 2.f * x;
        fC1_y    = -2.f * y;
        fS1_x    = 2.f * y;
        fS1_y    = 2.f * x;
        pSH6_z   = 2.f * 0.9461746957575601f * z;
        pSH7_x   = fTmp0B;
        pSH7_z   = fTmp0B_z * x;
        pSH5_y   = fTmp0B;
        pSH5_z   = fTmp0B_z * y;
        pSH8_x   = 0.5462742152960395f * fC1_x;
        pSH8_y   = 0.5462742152960395f * fC1_y;
        pSH4_x   = 0.5462742152960395f * fS1_x;
        pSH4_y   = 0.5462742152960395f * fS1_y;

        v_x += v_colors_local
             * (pSH4_x * static_cast<opmath_t>(coeffs[3 * D + c])
                + pSH8_x * static_cast<opmath_t>(coeffs[7 * D + c])
                + pSH7_x * static_cast<opmath_t>(coeffs[6 * D + c]));
        v_y += v_colors_local
             * (pSH4_y * static_cast<opmath_t>(coeffs[3 * D + c])
                + pSH8_y * static_cast<opmath_t>(coeffs[7 * D + c])
                + pSH5_y * static_cast<opmath_t>(coeffs[4 * D + c]));
        v_z += v_colors_local
             * (pSH6_z * static_cast<opmath_t>(coeffs[5 * D + c])
                + pSH7_z * static_cast<opmath_t>(coeffs[6 * D + c])
                + pSH5_z * static_cast<opmath_t>(coeffs[4 * D + c]));
    }

    if(degree < 3)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0C  = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B  = 1.445305721320277f * z;
    float fC2     = x * fC1 - y * fS1;
    float fS2     = x * fS1 + y * fC1;
    float pSH12   = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13   = fTmp0C * x;
    float pSH11   = fTmp0C * y;
    float pSH14   = fTmp1B * fC1;
    float pSH10   = fTmp1B * fS1;
    float pSH15   = -0.5900435899266435f * fC2;
    float pSH9    = -0.5900435899266435f * fS2;
    acc[8]       += pSH9 * v_colors_local;
    acc[9]       += pSH10 * v_colors_local;
    acc[10]      += pSH11 * v_colors_local;
    acc[11]      += pSH12 * v_colors_local;
    acc[12]      += pSH13 * v_colors_local;
    acc[13]      += pSH14 * v_colors_local;
    acc[14]      += pSH15 * v_colors_local;

    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x, pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y,
        pSH14_z, pSH10_x, pSH10_y, pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if(v_dir != nullptr)
    {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x    = fC1 + x * fC1_x - y * fS1_x;
        fC2_y    = x * fC1_y - fS1 - y * fS1_y;
        fS2_x    = fS1 + x * fS1_x + y * fC1_x;
        fS2_y    = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z  = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x  = fTmp0C;
        pSH13_z  = fTmp0C_z * x;
        pSH11_y  = fTmp0C;
        pSH11_z  = fTmp0C_z * y;
        pSH14_x  = fTmp1B * fC1_x;
        pSH14_y  = fTmp1B * fC1_y;
        pSH14_z  = fTmp1B_z * fC1;
        pSH10_x  = fTmp1B * fS1_x;
        pSH10_y  = fTmp1B * fS1_y;
        pSH10_z  = fTmp1B_z * fS1;
        pSH15_x  = -0.5900435899266435f * fC2_x;
        pSH15_y  = -0.5900435899266435f * fC2_y;
        pSH9_x   = -0.5900435899266435f * fS2_x;
        pSH9_y   = -0.5900435899266435f * fS2_y;

        v_x += v_colors_local
             * (pSH9_x * static_cast<opmath_t>(coeffs[8 * D + c])
                + pSH15_x * static_cast<opmath_t>(coeffs[14 * D + c])
                + pSH10_x * static_cast<opmath_t>(coeffs[9 * D + c])
                + pSH14_x * static_cast<opmath_t>(coeffs[13 * D + c])
                + pSH13_x * static_cast<opmath_t>(coeffs[12 * D + c]));

        v_y += v_colors_local
             * (pSH9_y * static_cast<opmath_t>(coeffs[8 * D + c])
                + pSH15_y * static_cast<opmath_t>(coeffs[14 * D + c])
                + pSH10_y * static_cast<opmath_t>(coeffs[9 * D + c])
                + pSH14_y * static_cast<opmath_t>(coeffs[13 * D + c])
                + pSH11_y * static_cast<opmath_t>(coeffs[10 * D + c]));

        v_z += v_colors_local
             * (pSH12_z * static_cast<opmath_t>(coeffs[11 * D + c])
                + pSH13_z * static_cast<opmath_t>(coeffs[12 * D + c])
                + pSH11_z * static_cast<opmath_t>(coeffs[10 * D + c])
                + pSH14_z * static_cast<opmath_t>(coeffs[13 * D + c])
                + pSH10_z * static_cast<opmath_t>(coeffs[9 * D + c]));
    }

    if(degree < 4)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0D  = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C  = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B  = -1.770130769779931f * z;
    float fC3     = x * fC2 - y * fS2;
    float fS3     = x * fS2 + y * fC2;
    float pSH20   = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21   = fTmp0D * x;
    float pSH19   = fTmp0D * y;
    float pSH22   = fTmp1C * fC1;
    float pSH18   = fTmp1C * fS1;
    float pSH23   = fTmp2B * fC2;
    float pSH17   = fTmp2B * fS2;
    float pSH24   = 0.6258357354491763f * fC3;
    float pSH16   = 0.6258357354491763f * fS3;
    acc[15]      += pSH16 * v_colors_local;
    acc[16]      += pSH17 * v_colors_local;
    acc[17]      += pSH18 * v_colors_local;
    acc[18]      += pSH19 * v_colors_local;
    acc[19]      += pSH20 * v_colors_local;
    acc[20]      += pSH21 * v_colors_local;
    acc[21]      += pSH22 * v_colors_local;
    acc[22]      += pSH23 * v_colors_local;
    acc[23]      += pSH24 * v_colors_local;

    float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z, pSH21_x, pSH21_z, pSH19_y, pSH19_z,
        pSH22_x, pSH22_y, pSH22_z, pSH18_x, pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if(v_dir != nullptr)
    {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x    = fC2 + x * fC2_x - y * fS2_x;
        fC3_y    = x * fC2_y - fS2 - y * fS2_y;
        fS3_x    = fS2 + y * fC2_x + x * fS2_x;
        fS3_y    = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z  = 1.984313483298443f * (pSH12 + z * pSH12_z) + -1.006230589874905f * pSH6_z;
        pSH21_x  = fTmp0D;
        pSH21_z  = fTmp0D_z * x;
        pSH19_y  = fTmp0D;
        pSH19_z  = fTmp0D_z * y;
        pSH22_x  = fTmp1C * fC1_x;
        pSH22_y  = fTmp1C * fC1_y;
        pSH22_z  = fTmp1C_z * fC1;
        pSH18_x  = fTmp1C * fS1_x;
        pSH18_y  = fTmp1C * fS1_y;
        pSH18_z  = fTmp1C_z * fS1;
        pSH23_x  = fTmp2B * fC2_x;
        pSH23_y  = fTmp2B * fC2_y;
        pSH23_z  = fTmp2B_z * fC2;
        pSH17_x  = fTmp2B * fS2_x;
        pSH17_y  = fTmp2B * fS2_y;
        pSH17_z  = fTmp2B_z * fS2;
        pSH24_x  = 0.6258357354491763f * fC3_x;
        pSH24_y  = 0.6258357354491763f * fC3_y;
        pSH16_x  = 0.6258357354491763f * fS3_x;
        pSH16_y  = 0.6258357354491763f * fS3_y;

        v_x += v_colors_local
             * (pSH16_x * static_cast<opmath_t>(coeffs[15 * D + c])
                + pSH24_x * static_cast<opmath_t>(coeffs[23 * D + c])
                + pSH17_x * static_cast<opmath_t>(coeffs[16 * D + c])
                + pSH23_x * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH18_x * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH22_x * static_cast<opmath_t>(coeffs[21 * D + c])
                + pSH21_x * static_cast<opmath_t>(coeffs[20 * D + c]));
        v_y += v_colors_local
             * (pSH16_y * static_cast<opmath_t>(coeffs[15 * D + c])
                + pSH24_y * static_cast<opmath_t>(coeffs[23 * D + c])
                + pSH17_y * static_cast<opmath_t>(coeffs[16 * D + c])
                + pSH23_y * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH18_y * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH22_y * static_cast<opmath_t>(coeffs[21 * D + c])
                + pSH19_y * static_cast<opmath_t>(coeffs[18 * D + c]));
        v_z += v_colors_local
             * (pSH20_z * static_cast<opmath_t>(coeffs[19 * D + c])
                + pSH21_z * static_cast<opmath_t>(coeffs[20 * D + c])
                + pSH19_z * static_cast<opmath_t>(coeffs[18 * D + c])
                + pSH22_z * static_cast<opmath_t>(coeffs[21 * D + c])
                + pSH18_z * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH23_z * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH17_z * static_cast<opmath_t>(coeffs[16 * D + c]));

        vec3 dir_n   = vec3(x, y, z);
        vec3 v_dir_n = vec3(v_x, v_y, v_z);
        vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

        v_dir->x = v_d.x;
        v_dir->y = v_d.y;
        v_dir->z = v_d.z;
    }
}

// Generic forward kernel: any K, one thread per (Gaussian, channel).
// scalar_t is fp16 or fp32; reads are upcast to opmath_t (fp32) inside the helper.
template<typename scalar_t, typename opmath_t>
__global__ void spherical_harmonics_l1_plus_fwd_kernel(
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ camera_offsets,
    const scalar_t *__restrict__ coeffs, // [N, K - 1, D]
    const bool *__restrict__ masks,      // [..., N]
    const int64_t *__restrict__ batch_ids,
    const int64_t *__restrict__ camera_ids,
    const int64_t *__restrict__ gaussian_ids,
    opmath_t *__restrict__ colors // [..., N, D]
)
{
    // parallelize over B * gaussian_count * D
    auto idx            = cg::this_grid().thread_rank();
    const bool packed   = batch_ids != nullptr;
    const int64_t count = (packed ? gaussian_count : static_cast<int64_t>(B) * C * gaussian_count) * D;
    if(idx >= count)
    {
        return;
    }
    const int64_t local_elem_id = idx / D;
    const int64_t image_id      = packed ? 0 : local_elem_id / gaussian_count;
    const int64_t output_id = packed ? local_elem_id : image_id * N + local_elem_id % gaussian_count + gaussian_offset;
    const int64_t batch_id  = packed ? batch_ids[output_id] : image_id / C;
    const int64_t camera_id = packed ? camera_ids[output_id] : image_id % C;
    const int64_t gaussian_id = packed ? gaussian_ids[output_id] : local_elem_id % gaussian_count + gaussian_offset;
    const int64_t coeff_id    = packed ? output_id : gaussian_id;
    const uint32_t c          = idx % D; // output channel
    if(masks != nullptr && !masks[output_id])
    {
        return;
    }
    if(degrees_to_use == 0)
    {
        colors[output_id * D + c] = 0;
        return;
    }
    const int64_t image_offset = batch_id * C + camera_id;
    const float *mean          = means + (batch_id * N + gaussian_id) * 3;
    const vec3 dir             = view_direction_from_camera_data(mean, viewmats, camera_offsets, image_offset);
    sh_l1_plus_coeffs_to_color_fast<scalar_t>(
        degrees_to_use, D, c, dir, coeffs + coeff_id * K * D, colors + output_id * D
    );
}

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
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    const uint32_t E = batch_ids.has_value() ? coeffs.size(0) : B * C * N;

    // parallelize over B * N * D
    int64_t n_elements             = static_cast<int64_t>(E) * D;
    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    at::Tensor camera_offsets;
    if(viewmats_rs.has_value())
    {
        camera_offsets = precompute_spherical_harmonics_camera_offsets(viewmats, viewmats_rs);
    }

    // Dispatch on the coeff dtype (fp16/fp32); colors use opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_harmonics_l1_plus_fwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t   = at::opmath_type<scalar_t>;
                auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                auto *colors_ptr = colors.data_ptr<opmath_t>();

                spherical_harmonics_l1_plus_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                    0,
                    batch_ids.has_value() ? E : N,
                    B,
                    C,
                    N,
                    K,
                    D,
                    degrees_to_use,
                    means.const_data_ptr<float>(),
                    viewmats.const_data_ptr<float>(),
                    camera_offsets.defined() ? camera_offsets.const_data_ptr<float>() : nullptr,
                    coeffs.const_data_ptr<scalar_t>(),
                    masks_ptr,
                    batch_ids.has_value() ? batch_ids.value().const_data_ptr<int64_t>() : nullptr,
                    camera_ids.has_value() ? camera_ids.value().const_data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                    colors_ptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        ),
        at::kFloat,
        at::kHalf
    );
}

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
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    TORCH_INTERNAL_ASSERT(!batch_ids.has_value() && !camera_ids.has_value() && !gaussian_ids.has_value());

    constexpr unsigned int threads = 256;

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);
        int64_t n_elements                        = static_cast<int64_t>(B) * C * gaussian_count * D;
        unsigned int blocks = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
        if(blocks > 0)
        {
            AT_DISPATCH_V2(
                coeffs.scalar_type(),
                "spherical_harmonics_l1_plus_fwd_kernel",
                AT_WRAP(
                    [&]()
                    {
                        using opmath_t   = at::opmath_type<scalar_t>;
                        auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *colors_ptr = colors.data_ptr<opmath_t>();

                        spherical_harmonics_l1_plus_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                            gaussian_offset,
                            gaussian_count,
                            B,
                            C,
                            N,
                            K,
                            D,
                            degrees_to_use,
                            means.const_data_ptr<float>(),
                            viewmats.const_data_ptr<float>(),
                            nullptr,
                            coeffs.const_data_ptr<scalar_t>(),
                            masks_ptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            colors_ptr
                        );
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                ),
                at::kFloat,
                at::kHalf
            );
        }
    }
    merge_streams();
}

template<typename scalar_t, typename opmath_t, int DEGREE>
__global__ void spherical_harmonics_l1_plus_bwd_kernel(
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ camera_offsets,
    const scalar_t *__restrict__ coeffs, // [N, K - 1, D]
    const bool *__restrict__ masks,      // [..., N]
    const int64_t *__restrict__ batch_ids,
    const int64_t *__restrict__ camera_ids,
    const int64_t *__restrict__ gaussian_ids,
    const opmath_t *__restrict__ v_colors, // [..., N, D]
    scalar_t *__restrict__ v_coeffs,       // [N, K - 1, D] (coeff dtype)
    float *__restrict__ v_means,
    float *__restrict__ v_viewdirs // [..., N, 3] optional
)
{
    // One thread per (coefficient row, channel). Dense mode reduces over all
    // batch-camera images in registers; packed mode has one coefficient row per
    // output element. Each thread writes its K results once in the coeff dtype.
    const int64_t idx   = cg::this_grid().thread_rank();
    const int64_t count = gaussian_count * static_cast<int64_t>(D);
    if(idx >= count)
    {
        return;
    }

    const bool packed      = batch_ids != nullptr;
    const int64_t coeff_id = idx / D + gaussian_offset;
    const uint32_t c       = static_cast<uint32_t>(idx % D);

    // The degree is a template parameter, so each kernel variant uses only the
    // register accumulator slots it needs (L1+ excludes the DC coefficient).
    constexpr int MAX_K = (DEGREE + 1) * (DEGREE + 1) - 1;

    if constexpr(DEGREE == 0)
    {
        // No l1+ coefficients contribute; write zeros so the output is defined.
        scalar_t *v_coeffs_ptr = v_coeffs + coeff_id * K * D;
#pragma unroll
        for(int k = 0; k < MAX_K; ++k)
        {
            if(static_cast<uint32_t>(k) < K)
            {
                v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(0.f);
            }
        }
        for(uint32_t k = MAX_K; k < K; ++k)
        {
            v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(0.f);
        }
        return;
    }

    std::array<opmath_t, MAX_K> acc{};

    const scalar_t *coeffs_ptr = coeffs + coeff_id * K * D;
    const uint32_t image_count = packed ? 1 : B * C;
    for(uint32_t dense_image_id = 0; dense_image_id < image_count; ++dense_image_id)
    {
        const int64_t output_id = packed ? coeff_id : static_cast<int64_t>(dense_image_id) * N + coeff_id;
        if(masks != nullptr && !masks[output_id])
        {
            continue;
        }

        const int64_t batch_id     = packed ? batch_ids[output_id] : dense_image_id / C;
        const int64_t camera_id    = packed ? camera_ids[output_id] : dense_image_id % C;
        const int64_t gaussian_id  = packed ? gaussian_ids[output_id] : coeff_id;
        const int64_t image_offset = batch_id * C + camera_id;
        const float *mean          = means + (batch_id * N + gaussian_id) * 3;
        const vec3 dir             = view_direction_from_camera_data(mean, viewmats, camera_offsets, image_offset);
        vec3 v_dir                 = {0.f, 0.f, 0.f};
        sh_l1_plus_coeffs_to_color_fast_vjp<scalar_t>(
            DEGREE,
            D,
            c,
            dir,
            coeffs_ptr,
            v_colors + output_id * D,
            acc.data(),
            v_means == nullptr && v_viewdirs == nullptr ? nullptr : &v_dir
        );

        // Adjacent threads handle channels of the same coefficient row. Combine
        // their direction gradients before touching the geometry outputs.
        if((v_means != nullptr || v_viewdirs != nullptr) && reduce_view_direction_channels(coeff_id, v_dir))
        {
            if(v_means != nullptr)
            {
                float *v_mean = v_means + (batch_id * N + gaussian_id) * 3;
                gpuAtomicAdd(v_mean, v_dir.x);
                gpuAtomicAdd(v_mean + 1, v_dir.y);
                gpuAtomicAdd(v_mean + 2, v_dir.z);
            }
            if(v_viewdirs != nullptr)
            {
                float *v_dir_out = v_viewdirs + output_id * 3;
                gpuAtomicAdd(v_dir_out, v_dir.x);
                gpuAtomicAdd(v_dir_out + 1, v_dir.y);
                gpuAtomicAdd(v_dir_out + 2, v_dir.z);
            }
        }
    }

    scalar_t *v_coeffs_ptr = v_coeffs + coeff_id * K * D;
#pragma unroll
    for(int k = 0; k < MAX_K; ++k)
    {
        if(static_cast<uint32_t>(k) < K)
        {
            v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(acc[k]);
        }
    }
    // Padded bases may allocate K > MAX_K; zero the tail so at::empty outputs are defined.
    for(uint32_t k = MAX_K; k < K; ++k)
    {
        v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(0.f);
    }
}

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
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    const uint32_t E = batch_ids.has_value() ? coeffs.size(0) : B * C * N;

    auto stream                 = at::cuda::getCurrentCUDAStream();
    const bool needs_v_viewdirs = v_viewmats.has_value() || v_viewmats_rs.has_value();

    // One thread per (coefficient row, channel). Dense mode loops over B * C
    // images internally; packed mode has one coefficient row per output.
    const int64_t coefficient_rows = batch_ids.has_value() ? E : N;
    int64_t n_elements             = coefficient_rows * D;
    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    at::Tensor v_viewdirs;
    if(needs_v_viewdirs)
    {
        auto v_viewdirs_sizes   = v_colors.sizes().vec();
        v_viewdirs_sizes.back() = 3;
        v_viewdirs              = at::zeros(v_viewdirs_sizes, means.options());
    }

    at::Tensor camera_offsets;
    if(viewmats_rs.has_value())
    {
        camera_offsets = precompute_spherical_harmonics_camera_offsets(viewmats, viewmats_rs);
    }

    // Dispatch on the coeff dtype (fp16/fp32). v_coeffs is written in the coeff
    // dtype after fp32 register accumulation; v_colors uses opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_harmonics_l1_plus_bwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t        = at::opmath_type<scalar_t>;
                auto *masks_ptr       = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                auto *v_colors_ptr    = v_colors.const_data_ptr<opmath_t>();
                const bool dispatched = dispatch::dispatch(
                    dispatch::IntParam<0, 1, 2, 3, 4>{static_cast<int>(degrees_to_use)},
                    [&]<typename DegConst>()
                    {
                        constexpr int DEGREE = DegConst::value;
                        spherical_harmonics_l1_plus_bwd_kernel<scalar_t, opmath_t, DEGREE>
                            <<<blocks, threads, 0, stream>>>(
                                0,
                                coefficient_rows,
                                B,
                                C,
                                N,
                                K,
                                D,
                                means.const_data_ptr<float>(),
                                viewmats.const_data_ptr<float>(),
                                camera_offsets.defined() ? camera_offsets.const_data_ptr<float>() : nullptr,
                                coeffs.const_data_ptr<scalar_t>(),
                                masks_ptr,
                                batch_ids.has_value() ? batch_ids.value().const_data_ptr<int64_t>() : nullptr,
                                camera_ids.has_value() ? camera_ids.value().const_data_ptr<int64_t>() : nullptr,
                                gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                                v_colors_ptr,
                                v_coeffs.data_ptr<scalar_t>(),
                                v_means.has_value() ? v_means.value().data_ptr<float>() : nullptr,
                                needs_v_viewdirs ? v_viewdirs.data_ptr<float>() : nullptr
                            );
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                );
                TORCH_CHECK(dispatched, "Unsupported SH degree: ", degrees_to_use);
            }
        ),
        at::kFloat,
        at::kHalf
    );

    if(needs_v_viewdirs)
    {
        launch_spherical_harmonics_view_direction_vjp_reduction<ViewmatGradientUpdate::Assign>(
            N, 0, N, viewmats, v_viewdirs, v_viewmats, viewmats_rs, batch_ids, camera_ids, v_viewmats_rs
        );
    }
}

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
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    TORCH_INTERNAL_ASSERT(!batch_ids.has_value() && !camera_ids.has_value() && !gaussian_ids.has_value());

    constexpr unsigned int threads = 256;

    if(N == 0)
    {
        return;
    }

    at::Tensor v_viewdirs;
    if(v_viewmats.has_value())
    {
        auto v_viewdirs_sizes   = v_colors.sizes().vec();
        v_viewdirs_sizes.back() = 3;
        v_viewdirs              = at::empty(v_viewdirs_sizes, means.options());
    }

    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventCreate(&events[device_id], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getStreamFromPool(false, device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[device_id]));

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);

        if(gaussian_count > 0)
        {
            std::vector<void *> prefetch_ptrs;
            std::vector<size_t> prefetch_sizes;
            auto append_prefetch_range
                = [&prefetch_ptrs,
                   &prefetch_sizes](const at::Tensor &tensor, int64_t row_offset, int64_t row_count, int64_t row_stride)
            {
                prefetch_ptrs.emplace_back(
                    static_cast<char *>(tensor.data_ptr()) + row_offset * row_stride * tensor.element_size()
                );
                prefetch_sizes.emplace_back(row_count * row_stride * tensor.element_size());
            };

            append_prefetch_range(v_coeffs, gaussian_offset, gaussian_count, v_coeffs.stride(-3));
            for(const auto batch_id: c10::irange(B))
            {
                const int64_t row_offset = static_cast<int64_t>(batch_id) * N + gaussian_offset;
                if(v_means.has_value())
                {
                    append_prefetch_range(v_means.value(), row_offset, gaussian_count, v_means.value().stride(-2));
                }
            }
            for(const auto image_id: c10::irange(B * C))
            {
                const int64_t row_offset = static_cast<int64_t>(image_id) * N + gaussian_offset;
                append_prefetch_range(v_colors, row_offset, gaussian_count, v_colors.stride(-2));
                if(v_viewmats.has_value())
                {
                    append_prefetch_range(v_viewdirs, row_offset, gaussian_count, v_viewdirs.stride(-2));
                }
            }

#if CUDART_VERSION < 13000
            for(const auto range_id: c10::irange(prefetch_ptrs.size()))
            {
                C10_CUDA_CHECK(
                    cudaMemPrefetchAsync(prefetch_ptrs[range_id], prefetch_sizes[range_id], device_id, stream)
                );
            }
#else
            const cudaMemLocation location                  = {cudaMemLocationTypeDevice, device_id};
            std::vector<cudaMemLocation> prefetch_locations = {location};
            std::vector<size_t> prefetch_location_indices   = {0};
            C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(
                prefetch_ptrs.data(),
                prefetch_sizes.data(),
                prefetch_ptrs.size(),
                prefetch_locations.data(),
                prefetch_location_indices.data(),
                prefetch_locations.size(),
                0,
                stream
            ));
#endif

            C10_CUDA_CHECK(cudaMemsetAsync(
                static_cast<char *>(v_coeffs.data_ptr())
                    + gaussian_offset * v_coeffs.stride(-3) * v_coeffs.element_size(),
                0,
                gaussian_count * v_coeffs.stride(-3) * v_coeffs.element_size(),
                stream
            ));
            if(v_means.has_value())
            {
                for(const auto batch_id: c10::irange(B))
                {
                    const int64_t row_offset = static_cast<int64_t>(batch_id) * N + gaussian_offset;
                    C10_CUDA_CHECK(cudaMemsetAsync(
                        static_cast<char *>(v_means.value().data_ptr())
                            + row_offset * v_means.value().stride(-2) * v_means.value().element_size(),
                        0,
                        gaussian_count * v_means.value().stride(-2) * v_means.value().element_size(),
                        stream
                    ));
                }
            }
            if(v_viewmats.has_value())
            {
                for(const auto image_id: c10::irange(B * C))
                {
                    const int64_t row_offset = static_cast<int64_t>(image_id) * N + gaussian_offset;
                    C10_CUDA_CHECK(cudaMemsetAsync(
                        static_cast<char *>(v_viewdirs.data_ptr())
                            + row_offset * v_viewdirs.stride(-2) * v_viewdirs.element_size(),
                        0,
                        gaussian_count * v_viewdirs.stride(-2) * v_viewdirs.element_size(),
                        stream
                    ));
                }
            }
        }
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[device_id]));
        C10_CUDA_CHECK(cudaEventDestroy(events[device_id]));

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);
        int64_t n_elements                        = gaussian_count * static_cast<int64_t>(D);
        unsigned int blocks = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
        if(blocks > 0)
        {
            AT_DISPATCH_V2(
                coeffs.scalar_type(),
                "spherical_harmonics_l1_plus_bwd_kernel",
                AT_WRAP(
                    [&]()
                    {
                        using opmath_t        = at::opmath_type<scalar_t>;
                        auto *masks_ptr       = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *v_colors_ptr    = v_colors.const_data_ptr<opmath_t>();
                        const bool dispatched = dispatch::dispatch(
                            dispatch::IntParam<0, 1, 2, 3, 4>{static_cast<int>(degrees_to_use)},
                            [&]<typename DegConst>()
                            {
                                constexpr int DEGREE = DegConst::value;
                                spherical_harmonics_l1_plus_bwd_kernel<scalar_t, opmath_t, DEGREE>
                                    <<<blocks, threads, 0, stream>>>(
                                        gaussian_offset,
                                        gaussian_count,
                                        B,
                                        C,
                                        N,
                                        K,
                                        D,
                                        means.const_data_ptr<float>(),
                                        viewmats.const_data_ptr<float>(),
                                        nullptr,
                                        coeffs.const_data_ptr<scalar_t>(),
                                        masks_ptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        v_colors_ptr,
                                        v_coeffs.data_ptr<scalar_t>(),
                                        v_means.has_value() ? v_means.value().data_ptr<float>() : nullptr,
                                        v_viewmats.has_value() ? v_viewdirs.data_ptr<float>() : nullptr
                                    );
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                            }
                        );
                        TORCH_CHECK(dispatched, "Unsupported SH degree: ", degrees_to_use);

                        if(v_viewmats.has_value())
                        {
                            launch_spherical_harmonics_view_direction_vjp_reduction<
                                ViewmatGradientUpdate::SystemAtomicAdd
                            >(N, gaussian_offset, gaussian_count, viewmats, v_viewdirs, v_viewmats);
                        }
                    }
                ),
                at::kFloat,
                at::kHalf
            );
        }
    }
    merge_streams();
}
// Backward of spherical_harmonics_l1_plus fused with the Adam update of shN.
// Dense mode only (one coefficient row per gaussian). Each thread owns one
// (gaussian, channel) column: it accumulates the shN gradient for that column in
// registers over all images, then applies torch.optim.Adam's update (no weight
// decay, no amsgrad) to the parameter and both moments in place, so the shN
// gradient is never written to memory. Rows with zero gradient are still
// updated, exactly as torch.optim.Adam does.
// SH_C0 * sh0, as in spherical_harmonics_l0.
constexpr float SH_L0_C0 = 0.2820947917738781f;
// Max channels for the fused color path (per-thread scratch indexed by channel).
constexpr uint32_t SH_FUSED_MAX_D = 4;

// sqrt.approx.f32 (what sqrtf compiles to under --use_fast_math), as an
// explicit instruction so its use does not depend on compiler flags.
__device__ __forceinline__ float __fsqrt_rn_approx(const float x)
{
    float y;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// One Adam element update (torch.optim.Adam, wd=0). Every fused / lazy path uses
// this single function so that replayed and eager updates are bitwise equal.
__device__ __forceinline__ void adam_elem(
    const float grad,
    float &p,
    float &m,
    float &v,
    const float beta1,
    const float beta2,
    const float eps,
    const float step_size,
    const float bias_correction2_sqrt
)
{
    // Explicit intrinsics: the compiler may not contract or reorder them (unlike
    // plain arithmetic under --use_fast_math), so every call site -- eager or
    // replayed -- computes bit-identical results. sqrt / division use the fast
    // hardware approximations (deterministic, ~1-2 ulp vs IEEE).
    m               = __fmaf_rn(beta1, m, __fmul_rn(__fsub_rn(1.f, beta1), grad));
    v               = __fmaf_rn(beta2, v, __fmul_rn(__fmul_rn(__fsub_rn(1.f, beta2), grad), grad));
    const float den = __fadd_rn(__fdividef(__fsqrt_rn_approx(v), bias_correction2_sqrt), eps);
    p               = __fsub_rn(p, __fdividef(__fmul_rn(step_size, m), den));
}

// COLORS = true: the forward was sh_colors_fused_fwd, i.e.
//   colors = max(C0 * sh0 + l1plus(shN) + 0.5, 0)
// so the color gradient is masked where the pre-activation is negative
// (recomputed exactly), and sh0 also gets its gradient and Adam step here.
template<int DEGREE, bool COLORS>
__global__ void spherical_harmonics_l1_plus_bwd_adam_kernel(
    const uint32_t BC,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t rows_per_block,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    float *__restrict__ coeffs,         // [N, K, D] updated in place
    const float *__restrict__ v_colors, // [B*C, N, D]
    float *__restrict__ exp_avg,        // [N, K, D]
    float *__restrict__ exp_avg_sq,     // [N, K, D]
    float *__restrict__ v_means,        // [N, 3] or nullptr
    const float step_size,              // lr / bias_correction1
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction2_sqrt,
    // COLORS only: sh0 [N, 1, D] and its Adam state / hyper-parameters
    float *__restrict__ sh0,
    float *__restrict__ sh0_exp_avg,
    float *__restrict__ sh0_exp_avg_sq,
    const float sh0_step_size,
    const float sh0_beta1,
    const float sh0_beta2,
    const float sh0_eps,
    const float sh0_bias_correction2_sqrt
)
{
    // Phase 1: one thread per (gaussian, channel) computes that column's shN
    // gradient (as in spherical_harmonics_l1_plus_bwd_kernel) and stages it in
    // shared memory. Phase 2: all threads sweep the block's contiguous
    // [rows_per_block, K, D] slab with coalesced Adam updates.
    extern __shared__ float grad_slab[]; // [rows_per_block * K * D]
    const uint32_t tid      = threadIdx.x;
    const int64_t row_begin = static_cast<int64_t>(blockIdx.x) * rows_per_block;
    const uint32_t rows     = static_cast<uint32_t>(N - row_begin < rows_per_block ? N - row_begin : rows_per_block);
    const uint32_t local    = tid / D;
    const uint32_t c        = tid % D;
    const int64_t coeff_id  = row_begin + local;

    constexpr int MAX_K = (DEGREE + 1) * (DEGREE + 1) - 1;
    float acc[MAX_K > 0 ? MAX_K : 1];
#pragma unroll
    for(int k = 0; k < (MAX_K > 0 ? MAX_K : 1); ++k)
    {
        acc[k] = 0.f;
    }

    float v_sh0 = 0.f; // COLORS only
    if(local < rows)
    {
        if constexpr(DEGREE > 0 || COLORS)
        {
            const float *coeffs_ptr = coeffs + coeff_id * K * D;
            for(uint32_t image_id = 0; image_id < BC; ++image_id)
            {
                const int64_t output_id = static_cast<int64_t>(image_id) * N + coeff_id;
                vec3 v_dir              = {0.f, 0.f, 0.f};
                float vc                = v_colors[output_id * D + c];
                if(vc != 0.f)
                {
                    const vec3 dir = view_direction_from_camera_data(means + coeff_id * 3, viewmats, nullptr, image_id);
                    if constexpr(COLORS)
                    {
                        // Recompute the pre-activation exactly as the forward did.
                        // torch's clamp_min backward passes the gradient only where
                        // x > 0 (zero at x == 0, which trained models hit often).
                        float l1[SH_FUSED_MAX_D];
                        l1[c] = 0.f;
                        if constexpr(DEGREE > 0)
                        {
                            sh_l1_plus_coeffs_to_color_fast<float>(DEGREE, D, c, dir, coeffs_ptr, l1);
                        }
                        const float x = (sh0[coeff_id * D + c] * SH_L0_C0 + l1[c]) + 0.5f;
                        if(!(x > 0.f))
                        {
                            vc = 0.f;
                        }
                        v_sh0 += vc * SH_L0_C0;
                    }
                    if constexpr(DEGREE > 0)
                    {
                        if(vc != 0.f)
                        {
                            float vloc[SH_FUSED_MAX_D];
                            vloc[c] = vc;
                            sh_l1_plus_coeffs_to_color_fast_vjp<float>(
                                DEGREE, D, c, dir, coeffs_ptr, vloc, acc, v_means == nullptr ? nullptr : &v_dir
                            );
                        }
                    }
                }
                if(v_means != nullptr && reduce_view_direction_channels(coeff_id, v_dir))
                {
                    float *v_mean = v_means + coeff_id * 3;
                    gpuAtomicAdd(v_mean, v_dir.x);
                    gpuAtomicAdd(v_mean + 1, v_dir.y);
                    gpuAtomicAdd(v_mean + 2, v_dir.z);
                }
            }
        }
#pragma unroll
        for(int k = 0; k < MAX_K; ++k)
        {
            if(static_cast<uint32_t>(k) < K)
                grad_slab[local * K * D + k * D + c] = acc[k];
        }
        for(uint32_t k = MAX_K; k < K; ++k)
        {
            grad_slab[local * K * D + k * D + c] = 0.f;
        }
    }
    __syncthreads();

    // Adam, matching ATen's fused Adam (ADAM_MODE::ORIGINAL, weight_decay=0).
    const uint32_t slab         = rows * K * D;
    const int64_t base          = row_begin * K * D;
    auto adam = [&](const float grad, float &p, float &m, float &v)
    { adam_elem(grad, p, m, v, beta1, beta2, eps, step_size, bias_correction2_sqrt); };
    // The launcher sizes blocks so that base and slab are multiples of 4 floats
    // whenever K * D * rows_per_block allows it; use float4 then.
    const bool vec4 = (base % 4 == 0) && (slab % 4 == 0);
    if(vec4)
    {
        float4 *p4 = reinterpret_cast<float4 *>(coeffs + base);
        float4 *m4 = reinterpret_cast<float4 *>(exp_avg + base);
        float4 *v4 = reinterpret_cast<float4 *>(exp_avg_sq + base);
        for(uint32_t e = tid; e < slab / 4; e += blockDim.x)
        {
            float4 p = p4[e], m = m4[e], v = v4[e];
            adam(grad_slab[4 * e + 0], p.x, m.x, v.x);
            adam(grad_slab[4 * e + 1], p.y, m.y, v.y);
            adam(grad_slab[4 * e + 2], p.z, m.z, v.z);
            adam(grad_slab[4 * e + 3], p.w, m.w, v.w);
            p4[e] = p;
            m4[e] = m;
            v4[e] = v;
        }
    }
    else
    {
        for(uint32_t e = tid; e < slab; e += blockDim.x)
        {
            const int64_t i = base + e;
            float p = coeffs[i], m = exp_avg[i], v = exp_avg_sq[i];
            adam(grad_slab[e], p, m, v);
            coeffs[i]     = p;
            exp_avg[i]    = m;
            exp_avg_sq[i] = v;
        }
    }

    if constexpr(COLORS)
    {
        // sh0 is [N, 1, D]: element (coeff_id, c) is owned by this thread and
        // consecutive threads touch consecutive floats.
        if(local < rows)
        {
            const int64_t i = coeff_id * D + c;
            float p = sh0[i], m = sh0_exp_avg[i], v = sh0_exp_avg_sq[i];
            adam_elem(v_sh0, p, m, v, sh0_beta1, sh0_beta2, sh0_eps, sh0_step_size, sh0_bias_correction2_sqrt);
            sh0[i]            = p;
            sh0_exp_avg[i]    = m;
            sh0_exp_avg_sq[i] = v;
        }
    }
}

// colors = max(C0 * sh0 + l1plus(shN) + 0.5, 0), dense [BC, N, D]. Same
// operation order as spherical_harmonics_l0 + spherical_harmonics_l1_plus
// followed by (+ 0.5).clamp_min(0), so the result is bitwise identical.
template<int DEGREE>
__global__ void sh_colors_fused_fwd_kernel(
    const uint32_t BC,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ sh0,    // [N, 1, D]
    const float *__restrict__ coeffs, // [N, K, D]
    float *__restrict__ colors        // [BC, N, D]
)
{
    const int64_t idx = cg::this_grid().thread_rank();
    if(idx >= static_cast<int64_t>(BC) * N * D)
    {
        return;
    }
    const uint32_t c        = static_cast<uint32_t>(idx % D);
    const int64_t output_id = idx / D;
    const uint32_t image_id = static_cast<uint32_t>(output_id / N);
    const int64_t coeff_id  = output_id % N;
    float l1[SH_FUSED_MAX_D];
    l1[c] = 0.f;
    if constexpr(DEGREE > 0)
    {
        const vec3 dir = view_direction_from_camera_data(means + coeff_id * 3, viewmats, nullptr, image_id);
        sh_l1_plus_coeffs_to_color_fast<float>(DEGREE, D, c, dir, coeffs + coeff_id * K * D, l1);
    }
    const float x = (sh0[coeff_id * D + c] * SH_L0_C0 + l1[c]) + 0.5f;
    colors[idx]   = x < 0.f ? 0.f : x;
}

void launch_sh_colors_fused_fwd_kernel(
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor sh0,
    const at::Tensor coeffs,
    at::Tensor colors
)
{
    const uint32_t D  = coeffs.size(-1);
    const uint32_t K  = coeffs.size(-2);
    const uint32_t N  = coeffs.size(0);
    const uint32_t BC = viewmats.numel() / 16;
    const int64_t n   = static_cast<int64_t>(BC) * N * D;
    if(n == 0)
    {
        return;
    }
    constexpr unsigned int threads = 256;
    const unsigned int blocks      = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n, threads));
    auto stream                    = at::cuda::getCurrentCUDAStream();
    const bool dispatched          = dispatch::dispatch(
        dispatch::IntParam<0, 1, 2, 3>{static_cast<int>(degrees_to_use)},
        [&]<typename DegConst>()
        {
            sh_colors_fused_fwd_kernel<DegConst::value><<<blocks, threads, 0, stream>>>(
                BC,
                N,
                K,
                D,
                means.const_data_ptr<float>(),
                viewmats.const_data_ptr<float>(),
                sh0.const_data_ptr<float>(),
                coeffs.const_data_ptr<float>(),
                colors.data_ptr<float>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    TORCH_CHECK(dispatched, "Unsupported SH degree: ", degrees_to_use);
}

void launch_spherical_harmonics_l1_plus_bwd_adam_kernel(
    const uint32_t degrees_to_use,
    const at::Tensor means,      // [N, 3]
    const at::Tensor viewmats,   // [B*C, 4, 4]
    at::Tensor coeffs,           // [N, K, D]
    const at::Tensor v_colors,   // [B*C, N, D]
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
)
{
    const uint32_t D  = coeffs.size(-1);
    const uint32_t K  = coeffs.size(-2);
    const uint32_t N  = coeffs.size(0);
    const uint32_t BC = viewmats.numel() / 16;
    const bool colors_mode = sh0.has_value();
    TORCH_CHECK(!colors_mode || D <= SH_FUSED_MAX_D, "fused SH colors support up to ", SH_FUSED_MAX_D, " channels");
    if(N == 0)
    {
        return;
    }
    TORCH_CHECK(D >= 1 && D <= 256, "fused SH Adam supports 1..256 channels");
    // Multiple of 4 rows keeps every block's slab 16-byte aligned for float4.
    const uint32_t rows_per_block = 256 / D >= 4 ? (256 / D) / 4 * 4 : 256 / D;
    const unsigned int threads    = rows_per_block * D;
    const unsigned int blocks     = (N + rows_per_block - 1) / rows_per_block;
    const size_t shmem            = static_cast<size_t>(rows_per_block) * K * D * sizeof(float);
    auto stream                   = at::cuda::getCurrentCUDAStream();
    const bool dispatched         = dispatch::dispatch(
        dispatch::IntParam<0, 1, 2, 3>{static_cast<int>(degrees_to_use)},
        [&]<typename DegConst>()
        {
            constexpr int DEGREE = DegConst::value;
            auto kernel          = colors_mode ? spherical_harmonics_l1_plus_bwd_adam_kernel<DEGREE, true>
                                               : spherical_harmonics_l1_plus_bwd_adam_kernel<DEGREE, false>;
            if(shmem > 48 * 1024)
            {
                C10_CUDA_CHECK(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem))
                );
            }
            kernel<<<blocks, threads, shmem, stream>>>(
                BC,
                N,
                K,
                D,
                rows_per_block,
                means.const_data_ptr<float>(),
                viewmats.const_data_ptr<float>(),
                coeffs.data_ptr<float>(),
                v_colors.const_data_ptr<float>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                v_means.has_value() ? v_means.value().data_ptr<float>() : nullptr,
                step_size,
                beta1,
                beta2,
                eps,
                bias_correction2_sqrt,
                colors_mode ? sh0.value().data_ptr<float>() : nullptr,
                colors_mode ? sh0_exp_avg.value().data_ptr<float>() : nullptr,
                colors_mode ? sh0_exp_avg_sq.value().data_ptr<float>() : nullptr,
                sh0_step_size,
                sh0_beta1,
                sh0_beta2,
                sh0_eps,
                sh0_bias_correction2_sqrt
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    TORCH_CHECK(dispatched, "Unsupported SH degree: ", degrees_to_use);
}
// ---------------------------------------------------------------------------
// Lazy (deferred) Adam for sh0 / shN.
//
// A zero-gradient Adam step is a deterministic function of (p, m, v) and that
// step's constants, and SH coefficients never influence which gaussians are
// visible. So a row whose color gradient is zero this step can skip its Adam
// update entirely; the missed steps are replayed later with exactly the same
// float operations (bitwise identical to having applied them eagerly):
//   - in the forward, in registers, for rows that are visible this step,
//   - in the backward, for rows that receive a gradient, before their update,
//   - for all rows in sh_lazy_catchup (before densification / eval / save).
// last[row] is the last optimizer step applied to the row (both sh0 and shN).
// Per-step constants live in tables indexed by step: tabs[0][t] = sh0
// step_size, tabs[1][t] = sh0 bias_correction2_sqrt, tabs[2]/[3] for shN.
// ---------------------------------------------------------------------------

struct LazyAdamConsts
{
    const float *step_size; // [T] indexed by step
    const float *bc2_sqrt;  // [T]
    float beta1, beta2, eps;
};

__device__ __forceinline__ void
lazy_adam_update(const float grad, float &p, float &m, float &v, const LazyAdamConsts &c, const int step)
{
    adam_elem(grad, p, m, v, c.beta1, c.beta2, c.eps, c.step_size[step], c.bc2_sqrt[step]);
}

// Apply zero-gradient steps (from, to] in place. `zero` is a runtime 0.f so the
// arithmetic is identical to an eager step with a zero gradient.
__device__ __forceinline__ void lazy_adam_replay(
    float &p, float &m, float &v, const LazyAdamConsts &c, const int from, const int to, const float zero
)
{
    for(int s = from + 1; s <= to; ++s)
    {
        lazy_adam_update(zero, p, m, v, c, s);
    }
}

// Replay only the moments (they do not depend on p): bitwise equal to the m, v
// produced by lazy_adam_replay.
__device__ __forceinline__ void
lazy_adam_replay_mv(float &m, float &v, const LazyAdamConsts &c, const int from, const int to, const float zero)
{
    float p = 0.f;
    for(int s = from + 1; s <= to; ++s)
    {
        lazy_adam_update(zero, p, m, v, c, s);
    }
}

// Both lazy kernels: a block owns rows_per_block consecutive rows (D threads
// per row in the per-row phases). Rows' shN slabs are read with coalesced
// sweeps, caught up, and staged in shared memory for the per-row math.

template<int DEGREE>
__global__ void sh_colors_lazy_fwd_kernel(
    const uint32_t BC,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t rows_per_block,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const bool *__restrict__ mask, // [BC, N] rows needed this step (superset of visible)
    const float *__restrict__ sh0,
    const float *__restrict__ sh0_m,
    const float *__restrict__ sh0_v,
    const float *__restrict__ shN,
    const float *__restrict__ shN_m,
    const float *__restrict__ shN_v,
    const int32_t *__restrict__ last,
    const int step, // the step being taken; parameters are current as of step - 1
    const LazyAdamConsts c0,
    const LazyAdamConsts cN,
    const float zero,
    float *__restrict__ colors // [BC, N, D]
)
{
    extern __shared__ float lazy_fwd_smem[];
    float *p_slab = lazy_fwd_smem;                                             // [rows_per_block * K * D]
    int *needed   = reinterpret_cast<int *>(p_slab + rows_per_block * K * D);  // [rows_per_block]
    int *row_from = needed + rows_per_block;                                   // [rows_per_block]

    const uint32_t tid      = threadIdx.x;
    const int64_t row_begin = static_cast<int64_t>(blockIdx.x) * rows_per_block;
    const uint32_t rows     = static_cast<uint32_t>(N - row_begin < rows_per_block ? N - row_begin : rows_per_block);
    const uint32_t local    = tid / D;
    const uint32_t c        = tid % D;
    const int64_t row       = row_begin + local;
    const int to            = step - 1;

    if(tid < rows)
    {
        bool any = false;
        for(uint32_t image_id = 0; image_id < BC; ++image_id)
        {
            any |= mask[static_cast<int64_t>(image_id) * N + row_begin + tid];
        }
        needed[tid]   = any;
        row_from[tid] = last[row_begin + tid];
    }
    __syncthreads();

    if constexpr(DEGREE > 0)
    {
        const uint32_t KD   = K * D;
        const uint32_t slab = rows * KD;
        const int64_t base  = row_begin * KD;
        for(uint32_t e = tid; e < slab; e += blockDim.x)
        {
            const uint32_t lr = e / KD;
            if(!needed[lr])
            {
                continue;
            }
            const int64_t i = base + e;
            float p         = shN[i];
            if(row_from[lr] < to)
            {
                float m = shN_m[i], v = shN_v[i];
                lazy_adam_replay(p, m, v, cN, row_from[lr], to, zero);
            }
            p_slab[e] = p;
        }
        __syncthreads();
    }

    if(local >= rows)
    {
        return;
    }
    float s0 = 0.f;
    if(needed[local])
    {
        s0 = sh0[row * D + c];
        if(row_from[local] < to)
        {
            float m = sh0_m[row * D + c], v = sh0_v[row * D + c];
            lazy_adam_replay(s0, m, v, c0, row_from[local], to, zero);
        }
    }
    for(uint32_t image_id = 0; image_id < BC; ++image_id)
    {
        const int64_t output_id = static_cast<int64_t>(image_id) * N + row;
        if(!mask[output_id])
        {
            colors[output_id * D + c] = 0.f; // never rasterized
            continue;
        }
        float l1[SH_FUSED_MAX_D];
        l1[c] = 0.f;
        if constexpr(DEGREE > 0)
        {
            const vec3 dir = view_direction_from_camera_data(means + row * 3, viewmats, nullptr, image_id);
            sh_l1_plus_coeffs_to_color_fast<float>(DEGREE, D, c, dir, p_slab + local * K * D, l1);
        }
        const float x             = (s0 * SH_L0_C0 + l1[c]) + 0.5f;
        colors[output_id * D + c] = x < 0.f ? 0.f : x;
    }
}

template<int DEGREE>
__global__ void sh_colors_lazy_bwd_adam_kernel(
    const uint32_t BC,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t rows_per_block,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ v_colors, // [BC, N, D]
    float *__restrict__ sh0,
    float *__restrict__ sh0_m,
    float *__restrict__ sh0_v,
    float *__restrict__ shN,
    float *__restrict__ shN_m,
    float *__restrict__ shN_v,
    int32_t *__restrict__ last,
    const int step,
    const LazyAdamConsts c0,
    const LazyAdamConsts cN,
    const float zero,
    float *__restrict__ v_means // [N, 3] or nullptr
)
{
    extern __shared__ float lazy_smem[];
    const uint32_t KD = K * D;
    float *p_slab     = lazy_smem;                                                 // caught-up shN, by active slot
    float *grad_slab  = p_slab + rows_per_block * KD;                              // shN gradient, by active slot
    int *active       = reinterpret_cast<int *>(grad_slab + rows_per_block * KD); // [rows_per_block] flags
    int *active_list  = active + rows_per_block;                                   // [rows_per_block] local rows
    int *row_from     = active_list + rows_per_block;                              // [rows_per_block] by slot
    int *n_active_sm  = row_from + rows_per_block;                                 // [1 + warps] counters

    const uint32_t tid      = threadIdx.x;
    // Row sweeps use whole warps only (lanes stride along a row); blockDim.x
    // need not be a multiple of 32.
    const uint32_t lane = tid % 32, warp_id = tid / 32, n_warps = blockDim.x / 32;
    const int64_t row_begin = static_cast<int64_t>(blockIdx.x) * rows_per_block;
    const uint32_t rows     = static_cast<uint32_t>(N - row_begin < rows_per_block ? N - row_begin : rows_per_block);
    const int to            = step - 1;

    if(tid < rows_per_block)
    {
        active[tid] = 0;
    }
    __syncthreads();
    // A row is updated this step iff any of its color gradients is nonzero;
    // otherwise its (zero-gradient) update is deferred. Coalesced over the
    // block's [rows, D] gradient slab.
    for(uint32_t e = tid; e < rows * D; e += blockDim.x)
    {
        bool any = false;
        for(uint32_t image_id = 0; image_id < BC; ++image_id)
        {
            any |= v_colors[(static_cast<int64_t>(image_id) * N + row_begin) * D + e] != 0.f;
        }
        if(any)
        {
            active[e / D] = 1;
        }
    }
    __syncthreads();
    // Compact the active rows of this block (order preserved).
    if(warp_id == 0)
    {
        int count = 0;
        for(uint32_t base_row = 0; base_row < rows; base_row += 32)
        {
            const uint32_t r   = base_row + lane;
            const bool f       = r < rows && active[r];
            const unsigned bal = __ballot_sync(0xffffffffu, f);
            if(f)
            {
                const int slot     = count + __popc(bal & ((1u << lane) - 1u));
                active_list[slot]  = static_cast<int>(r);
                row_from[slot]     = last[row_begin + r];
            }
            count += __popc(bal);
        }
        if(lane == 0)
        {
            n_active_sm[0] = count;
        }
    }
    __syncthreads();
    const int n_active = n_active_sm[0];
    if(n_active == 0)
    {
        return;
    }

    // Stage caught-up shN of active rows: one warp per row, lanes along the row.
    for(int a = warp_id; warp_id < n_warps && a < n_active; a += n_warps)
    {
        const int64_t base = (row_begin + active_list[a]) * KD;
        const int from     = row_from[a];
        for(uint32_t k = lane; k < KD; k += 32)
        {
            float p = shN[base + k];
            if(from < to)
            {
                float m = shN_m[base + k], v = shN_v[base + k];
                lazy_adam_replay(p, m, v, cN, from, to, zero);
            }
            p_slab[a * KD + k] = p;
        }
    }
    __syncthreads();

    // Per (active row, channel) gradient math on contiguous threads.
    if(tid < static_cast<uint32_t>(n_active) * D)
    {
        const int a          = tid / D;
        const uint32_t c     = tid % D;
        const int64_t row    = row_begin + active_list[a];
        const int from       = row_from[a];
        constexpr int MAX_K  = (DEGREE + 1) * (DEGREE + 1) - 1;
        float acc[MAX_K > 0 ? MAX_K : 1];
#pragma unroll
        for(int k = 0; k < (MAX_K > 0 ? MAX_K : 1); ++k)
        {
            acc[k] = 0.f;
        }
        float s0 = sh0[row * D + c], s0_m = sh0_m[row * D + c], s0_v = sh0_v[row * D + c];
        lazy_adam_replay(s0, s0_m, s0_v, c0, from, to, zero);
        const float *ce = p_slab + a * KD;
        float v_sh0     = 0.f;
        for(uint32_t image_id = 0; image_id < BC; ++image_id)
        {
            const int64_t output_id = static_cast<int64_t>(image_id) * N + row;
            vec3 v_dir              = {0.f, 0.f, 0.f};
            float vc                = v_colors[output_id * D + c];
            if(vc != 0.f)
            {
                const vec3 dir = view_direction_from_camera_data(means + row * 3, viewmats, nullptr, image_id);
                float l1[SH_FUSED_MAX_D];
                l1[c] = 0.f;
                if constexpr(DEGREE > 0)
                {
                    sh_l1_plus_coeffs_to_color_fast<float>(DEGREE, D, c, dir, ce, l1);
                }
                const float x = (s0 * SH_L0_C0 + l1[c]) + 0.5f;
                if(!(x > 0.f)) // torch clamp_min backward
                {
                    vc = 0.f;
                }
                v_sh0 += vc * SH_L0_C0;
                if constexpr(DEGREE > 0)
                {
                    if(vc != 0.f)
                    {
                        float vloc[SH_FUSED_MAX_D];
                        vloc[c] = vc;
                        sh_l1_plus_coeffs_to_color_fast_vjp<float>(
                            DEGREE, D, c, dir, ce, vloc, acc, v_means == nullptr ? nullptr : &v_dir
                        );
                    }
                }
            }
            if(v_means != nullptr && reduce_view_direction_channels(row, v_dir))
            {
                float *v_mean = v_means + row * 3;
                gpuAtomicAdd(v_mean, v_dir.x);
                gpuAtomicAdd(v_mean + 1, v_dir.y);
                gpuAtomicAdd(v_mean + 2, v_dir.z);
            }
        }
#pragma unroll
        for(int k = 0; k < MAX_K; ++k)
        {
            if(static_cast<uint32_t>(k) < K)
                grad_slab[a * KD + k * D + c] = acc[k];
        }
        for(uint32_t k = MAX_K; k < K; ++k)
        {
            grad_slab[a * KD + k * D + c] = 0.f;
        }
        lazy_adam_update(v_sh0, s0, s0_m, s0_v, c0, step);
        sh0[row * D + c]   = s0;
        sh0_m[row * D + c] = s0_m;
        sh0_v[row * D + c] = s0_v;
    }
    __syncthreads();

    // shN update of active rows: one warp per row. p comes caught-up from
    // shared memory; the moments are re-read and replayed.
    for(int a = warp_id; warp_id < n_warps && a < n_active; a += n_warps)
    {
        const int64_t base = (row_begin + active_list[a]) * KD;
        const int from     = row_from[a];
        for(uint32_t k = lane; k < KD; k += 32)
        {
            float p = p_slab[a * KD + k], m = shN_m[base + k], v = shN_v[base + k];
            lazy_adam_replay_mv(m, v, cN, from, to, zero);
            lazy_adam_update(grad_slab[a * KD + k], p, m, v, cN, step);
            shN[base + k]   = p;
            shN_m[base + k] = m;
            shN_v[base + k] = v;
        }
    }
    if(tid < static_cast<uint32_t>(n_active))
    {
        last[row_begin + active_list[tid]] = step;
    }
}

// Replay all deferred steps up to and including `step` for every row.
__global__ void sh_lazy_catchup_kernel(
    const int64_t n_elems, // N * KD
    const uint32_t KD,
    float *__restrict__ p,
    float *__restrict__ m,
    float *__restrict__ v,
    const int32_t *__restrict__ last,
    const int step,
    const LazyAdamConsts consts,
    const float zero
)
{
    const int64_t i = cg::this_grid().thread_rank();
    if(i >= n_elems)
    {
        return;
    }
    const int from = last[i / KD];
    if(from >= step)
    {
        return;
    }
    float pp = p[i], mm = m[i], vv = v[i];
    lazy_adam_replay(pp, mm, vv, consts, from, step, zero);
    p[i] = pp;
    m[i] = mm;
    v[i] = vv;
}

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
)
{
    const uint32_t D  = shN.size(-1);
    const uint32_t K  = shN.size(-2);
    const uint32_t N  = shN.size(0);
    const uint32_t BC = viewmats.numel() / 16;
    const int64_t n   = static_cast<int64_t>(BC) * N * D;
    if(n == 0)
    {
        return;
    }
    const float *t                = tabs.const_data_ptr<float>();
    const int64_t T               = tabs.size(1);
    const LazyAdamConsts c0       = {t, t + T, sh0_beta1, sh0_beta2, sh0_eps};
    const LazyAdamConsts cN       = {t + 2 * T, t + 3 * T, shN_beta1, shN_beta2, shN_eps};
    const uint32_t rows_per_block = 256 / D >= 4 ? (256 / D) / 4 * 4 : 256 / D;
    const unsigned int threads    = rows_per_block * D;
    const unsigned int blocks     = (N + rows_per_block - 1) / rows_per_block;
    const size_t shmem = static_cast<size_t>(rows_per_block) * K * D * sizeof(float) + 2 * rows_per_block * sizeof(int);
    auto stream        = at::cuda::getCurrentCUDAStream();
    const bool dispatched = dispatch::dispatch(
        dispatch::IntParam<0, 1, 2, 3>{static_cast<int>(degrees_to_use)},
        [&]<typename DegConst>()
        {
            auto kernel = sh_colors_lazy_fwd_kernel<DegConst::value>;
            if(shmem > 48 * 1024)
            {
                C10_CUDA_CHECK(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem))
                );
            }
            kernel<<<blocks, threads, shmem, stream>>>(
                BC,
                N,
                K,
                D,
                rows_per_block,
                means.const_data_ptr<float>(),
                viewmats.const_data_ptr<float>(),
                mask.const_data_ptr<bool>(),
                sh0.const_data_ptr<float>(),
                sh0_m.const_data_ptr<float>(),
                sh0_v.const_data_ptr<float>(),
                shN.const_data_ptr<float>(),
                shN_m.const_data_ptr<float>(),
                shN_v.const_data_ptr<float>(),
                last.const_data_ptr<int32_t>(),
                step,
                c0,
                cN,
                0.f,
                colors.data_ptr<float>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    TORCH_CHECK(dispatched, "Unsupported SH degree: ", degrees_to_use);
}

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
)
{
    const uint32_t D  = shN.size(-1);
    const uint32_t K  = shN.size(-2);
    const uint32_t N  = shN.size(0);
    const uint32_t BC = viewmats.numel() / 16;
    if(N == 0)
    {
        return;
    }
    const float *t                = tabs.const_data_ptr<float>();
    const int64_t T               = tabs.size(1);
    const LazyAdamConsts c0       = {t, t + T, sh0_beta1, sh0_beta2, sh0_eps};
    const LazyAdamConsts cN       = {t + 2 * T, t + 3 * T, shN_beta1, shN_beta2, shN_eps};
    const uint32_t rows_per_block = 256 / D >= 4 ? (256 / D) / 4 * 4 : 256 / D;
    const unsigned int threads    = rows_per_block * D;
    const unsigned int blocks     = (N + rows_per_block - 1) / rows_per_block;
    const size_t shmem = 2 * static_cast<size_t>(rows_per_block) * K * D * sizeof(float)
                       + (3 * rows_per_block + 1 + 32) * sizeof(int);
    auto stream        = at::cuda::getCurrentCUDAStream();
    const bool dispatched = dispatch::dispatch(
        dispatch::IntParam<0, 1, 2, 3>{static_cast<int>(degrees_to_use)},
        [&]<typename DegConst>()
        {
            auto kernel = sh_colors_lazy_bwd_adam_kernel<DegConst::value>;
            if(shmem > 48 * 1024)
            {
                C10_CUDA_CHECK(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem))
                );
            }
            kernel<<<blocks, threads, shmem, stream>>>(
                BC,
                N,
                K,
                D,
                rows_per_block,
                means.const_data_ptr<float>(),
                viewmats.const_data_ptr<float>(),
                v_colors.const_data_ptr<float>(),
                sh0.data_ptr<float>(),
                sh0_m.data_ptr<float>(),
                sh0_v.data_ptr<float>(),
                shN.data_ptr<float>(),
                shN_m.data_ptr<float>(),
                shN_v.data_ptr<float>(),
                last.data_ptr<int32_t>(),
                step,
                c0,
                cN,
                0.f,
                v_means.has_value() ? v_means.value().data_ptr<float>() : nullptr
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    TORCH_CHECK(dispatched, "Unsupported SH degree: ", degrees_to_use);
}

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
)
{
    const int64_t n = p.numel();
    if(n == 0)
    {
        return;
    }
    const uint32_t KD             = static_cast<uint32_t>(n / p.size(0));
    const LazyAdamConsts consts   = {step_size_tab.const_data_ptr<float>(), bc2_sqrt_tab.const_data_ptr<float>(), beta1, beta2, eps};
    constexpr unsigned int threads = 256;
    const unsigned int blocks      = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n, threads));
    sh_lazy_catchup_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n,
        KD,
        p.data_ptr<float>(),
        m.data_ptr<float>(),
        v.data_ptr<float>(),
        last.const_data_ptr<int32_t>(),
        step,
        consts,
        0.f
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace gsplat
