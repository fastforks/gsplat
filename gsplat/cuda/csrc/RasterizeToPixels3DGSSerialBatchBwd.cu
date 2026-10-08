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

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <ATen/cuda/Atomic.cuh>
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>

#    include "Common.h"
#    include "Rasterization.h"
#    include "RasterizeToPixels3DGSDevice.cuh"
#    include "Utils.cuh"
#    include "Dispatch.h"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;
#    ifndef BWD_PPT
#        define BWD_PPT 2
#    endif

template<uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const int64_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t block_offset,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,       // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities // [..., N] or [nnz]
)
{
    auto block              = cg::this_thread_block();
    auto linear_block_index = block.group_index().x + block_offset;
    dim3 block_index(
        linear_block_index / (tile_width * tile_height),
        (linear_block_index / tile_width) % tile_height,
        linear_block_index % tile_width
    );

    uint32_t image_id = block_index.x;
    uint32_t tile_id  = block_index.y * tile_width + block_index.z;
    // Map each warp to an 8x4 pixel patch of a 16x16 tile instead of a 16x2
    // strip: gaussians are small and roughly isotropic, so a squarer patch
    // reduces the (warp, gaussian) pairs that need the gradient reduction and
    // raises the number of contributing lanes in each. Every pixel still has
    // one thread.
    uint32_t lx = block.thread_index().x, ly = block.thread_index().y;
    if(tile_size == 16)
    {
        constexpr uint32_t WW = 8, WH = 32 / WW;
        const uint32_t r      = block.thread_rank();
        const uint32_t w      = r / 32, lane = r % 32;
        lx                    = (w % (16 / WW)) * WW + lane % WW;
        ly                    = (w / (16 / WW)) * WH + lane / WW;
    }
    uint32_t i = block_index.y * tile_size + ly;
    uint32_t j = block_index.z * tile_size + lx;

    tile_offsets    += image_id * tile_height * tile_width;
    render_alphas   += image_id * image_height * image_width;
    last_ids        += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if(masks != nullptr && !masks[tile_id])
    {
        return;
    }

    const float px       = (float)j + 0.5f;
    const float py       = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size       = block.size();
    const int64_t num_intersections = range_end - range_start;
    const int64_t num_batches       = (num_intersections + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;                                            // [block_size]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);         // [block_size]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *rgbs_batch      = (float *)&conic_batch[block_size];                       // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final           = 1.0f - render_alphas[pix_id];
    float T                 = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM]      = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#    pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final   = cg::reduce(warp, bin_final, cg::greater<int>());
    for(int64_t b = 0; b < num_batches; ++b)
    {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch_end_offset is relative to this tile's first intersection.
        const int64_t batch_end_offset = num_intersections - 1 - block_size * b;
        const int64_t remaining        = batch_end_offset + 1;
        const uint32_t batch_size      = static_cast<uint32_t>(remaining < block_size ? remaining : block_size);
        const int64_t idx              = range_start + batch_end_offset - tr;
        if(idx >= range_start)
        {
            int32_t g            = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr]         = g;
            const vec2 xy        = means2d[g];
            const float opac     = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr]      = conics[g];
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        const int64_t first_t64 = batch_end_offset > warp_bin_final ? batch_end_offset - warp_bin_final : 0;
        const uint32_t first_t  = first_t64 < batch_size ? static_cast<uint32_t>(first_t64) : batch_size;
        for(uint32_t t = first_t; t < batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end_offset - t > bin_final)
            {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;

            if(valid)
            {
                conic                   = conic_batch[t];
                vec3 xy_opac            = xy_opacity_batch[t];
                opac                    = xy_opac.z;
                delta                   = {xy_opac.x - px, xy_opac.y - py};
                const GaussianWeight gw = eval_gaussian_weight(conic, delta.x, delta.y, opac);
                vis                     = gw.vis;
                alpha                   = gw.alpha;
                if(!gw.valid)
                {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid))
            {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            vec2 v_xy_abs_local     = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if(valid)
            {
                // compute the current T for this gaussian
                float ra         = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T               *= ra;
                // update v_rgb for this gaussian
                const float fac  = alpha * T;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if(backgrounds != nullptr)
                {
                    float accum = 0.f;
#    pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if(opac * vis <= MAX_ALPHA)
                {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local
                        = {0.5f * v_sigma * delta.x * delta.x,
                           v_sigma * delta.x * delta.y,
                           0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local
                        = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                           v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    if(v_means2d_abs != nullptr)
                    {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            bool reduced = false;
            // The reduce-scatter needs full 32-lane warps; tile_size=4 launches
            // 16-thread blocks and takes the per-component path below.
            if constexpr(CDIM + 8 <= 16)
            if(block.size() % 32 == 0)
            {
                reduced = true;
                // Reduce all gradient components at once; lane 2k ends up
                // holding component k and issues its atomic in parallel.
                // Layout: rgb[CDIM], conic[3], xy[2], xy_abs[2], opacity.
                float vals[16];
#    pragma unroll
                for(uint32_t k = 0; k < 16; ++k)
                {
                    vals[k] = 0.f;
                }
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    vals[k] = v_rgb_local[k];
                }
                vals[CDIM + 0]      = v_conic_local.x;
                vals[CDIM + 1]      = v_conic_local.y;
                vals[CDIM + 2]      = v_conic_local.z;
                vals[CDIM + 3]      = v_xy_local.x;
                vals[CDIM + 4]      = v_xy_local.y;
                vals[CDIM + 5]      = v_xy_abs_local.x;
                vals[CDIM + 6]      = v_xy_abs_local.y;
                vals[CDIM + 7]      = v_opacity_local;
                const uint32_t lane = warp.thread_rank();
                const float sum     = warpReduceScatter16(vals, lane);
                const uint32_t c    = lane >> 1;
                if((lane & 1) == 0 && c < CDIM + 8)
                {
                    const int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                    float *ptr      = nullptr;
                    if(c < CDIM)
                        ptr = v_colors + CDIM * g + c;
                    else if(c < CDIM + 3)
                        ptr = (float *)(v_conics) + 3 * g + (c - CDIM);
                    else if(c < CDIM + 5)
                        ptr = (float *)(v_means2d) + 2 * g + (c - CDIM - 3);
                    else if(c < CDIM + 7)
                        ptr = v_means2d_abs == nullptr ? nullptr : (float *)(v_means2d_abs) + 2 * g + (c - CDIM - 5);
                    else
                        ptr = v_opacities + g;
                    if(ptr != nullptr)
                        atomicAdd_system(ptr, sum);
                }
            }
            if(!reduced)
            {
                warpSum<CDIM>(v_rgb_local, warp);
                warpSum(v_conic_local, warp);
                warpSum(v_xy_local, warp);
                if(v_means2d_abs != nullptr)
                {
                    warpSum(v_xy_abs_local, warp);
                }
                warpSum(v_opacity_local, warp);
                if(warp.thread_rank() == 0)
                {
                    int32_t g        = id_batch[t]; // flatten index in [I * N] or [nnz]
                    float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
    #    pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        atomicAdd_system(v_rgb_ptr + k, v_rgb_local[k]);
                    }

                    float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                    atomicAdd_system(v_conic_ptr, v_conic_local.x);
                    atomicAdd_system(v_conic_ptr + 1, v_conic_local.y);
                    atomicAdd_system(v_conic_ptr + 2, v_conic_local.z);

                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    atomicAdd_system(v_xy_ptr, v_xy_local.x);
                    atomicAdd_system(v_xy_ptr + 1, v_xy_local.y);

                    if(v_means2d_abs != nullptr)
                    {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        atomicAdd_system(v_xy_abs_ptr, v_xy_abs_local.x);
                        atomicAdd_system(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }

                    atomicAdd_system(v_opacities + g, v_opacity_local);
                }
            }
        }
    }
}

// Backward with PPT pixels per thread for 16x16 tiles (128 threads for PPT=2).
// Each warp covers an 8x8 pixel patch (lane -> 8x4 sub-patch, pixel p adds
// 4 * p rows). A thread first sums its pixels' contributions to a gaussian's
// gradient in registers, so the warp reduction, the global atomics and the
// per-gaussian loop overhead are amortized over 64 pixels instead of 32.
// Per-pixel math is identical to rasterize_to_pixels_3dgs_bwd_kernel.
// 8 blocks/SM (<= 64 registers) was marginally faster than unconstrained.
template<uint32_t CDIM, uint32_t PPT>
__global__ void __launch_bounds__(256 / PPT, 8) rasterize_to_pixels_3dgs_bwd_ppt_kernel(
    const uint32_t I,
    const int64_t n_isects,
    const vec2 *__restrict__ means2d,
    const vec3 *__restrict__ conics,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ backgrounds,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas,
    vec2 *__restrict__ v_means2d_abs,
    vec2 *__restrict__ v_means2d,
    vec3 *__restrict__ v_conics,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities
)
{
    static_assert(CDIM + 8 <= 16, "reduce-scatter path needs CDIM + 8 <= 16");
    constexpr uint32_t TILE = 16, BLOCK = TILE * TILE / PPT;
    static_assert(PPT == 2 || PPT == 4, "patch mapping supports PPT 2 (8x8 per warp) or 4 (8x16 per warp)");
    auto block                 = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t linear_block_index = blockIdx.x;
    const uint32_t image_id = linear_block_index / (tile_width * tile_height);
    const uint32_t tile_id  = linear_block_index % (tile_width * tile_height);
    const uint32_t tile_y   = tile_id / tile_width, tile_x = tile_id % tile_width;

    tile_offsets    += image_id * tile_height * tile_width;
    render_alphas   += image_id * image_height * image_width;
    last_ids        += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
        if(!masks[tile_id])
        {
            return;
        }
    }

    const uint32_t tr   = block.thread_rank();
    const uint32_t lane = tr % 32, w = tr / 32; // warps tile the 16x16 tile in 8 x (4 * PPT) patches
    const uint32_t lx   = (w % 2) * 8 + lane % 8;
    const uint32_t ly0  = (w / 2) * (4 * PPT) + lane / 8; // pixel p at row ly0 + 4 * p

    float px[PPT], py[PPT], T_final[PPT], T[PPT], buffer[PPT][CDIM], v_render_c[PPT][CDIM], v_render_a[PPT];
    int32_t bin_final[PPT];
    bool inside[PPT];
    int32_t warp_bin_final = 0;
#    pragma unroll
    for(uint32_t p = 0; p < PPT; ++p)
    {
        const uint32_t i = tile_y * TILE + ly0 + 4 * p;
        const uint32_t j = tile_x * TILE + lx;
        px[p]            = (float)j + 0.5f;
        py[p]            = (float)i + 0.5f;
        inside[p]        = (i < image_height && j < image_width);
        const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);
        T_final[p]       = 1.0f - render_alphas[pix_id];
        T[p]             = T_final[p];
        bin_final[p]     = inside[p] ? last_ids[pix_id] : 0;
#    pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            buffer[p][k]     = 0.f;
            v_render_c[p][k] = v_render_colors[pix_id * CDIM + k];
        }
        v_render_a[p] = v_render_alphas[pix_id];
        warp_bin_final = max(warp_bin_final, bin_final[p]);
    }
    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int64_t num_intersections = range_end - range_start;
    const int64_t num_batches       = (num_intersections + BLOCK - 1) / BLOCK;

    __shared__ int32_t id_batch[BLOCK];
    // Per-gaussian data packed for wide shared-memory loads:
    // pk0 = (x, y, opacity, conic.x), pk1 = (conic.y, conic.z, rgb0, rgb1).
    __shared__ float4 pk0_batch[BLOCK];
    __shared__ float4 pk1_batch[BLOCK];
    __shared__ float rgbs_batch[BLOCK * CDIM];

    float bg_dot[PPT];
#    pragma unroll
    for(uint32_t p = 0; p < PPT; ++p)
    {
        float accum = 0.f;
        if(backgrounds != nullptr)
        {
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                accum += backgrounds[k] * v_render_c[p][k];
            }
        }
        bg_dot[p] = accum;
    }

    for(int64_t b = 0; b < num_batches; ++b)
    {
        block.sync();
        const int64_t batch_end_offset = num_intersections - 1 - BLOCK * b;
        const int64_t remaining        = batch_end_offset + 1;
        const uint32_t batch_size      = static_cast<uint32_t>(remaining < BLOCK ? remaining : BLOCK);
        const int64_t idx              = range_start + batch_end_offset - tr;
        if(idx >= range_start)
        {
            const int32_t g = flatten_ids[idx];
            id_batch[tr]    = g;
            const vec2 xy   = means2d[g];
            const vec3 cn   = conics[g];
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
            pk0_batch[tr] = make_float4(xy.x, xy.y, opacities[g], cn.x);
            pk1_batch[tr] = make_float4(cn.y, cn.z, colors[g * CDIM], CDIM > 1 ? colors[g * CDIM + 1] : 0.f);
        }
        block.sync();
        const int64_t first_t64 = batch_end_offset > warp_bin_final ? batch_end_offset - warp_bin_final : 0;
        const uint32_t first_t  = first_t64 < batch_size ? static_cast<uint32_t>(first_t64) : batch_size;
        for(uint32_t t = first_t; t < batch_size; ++t)
        {
            const float4 a0    = pk0_batch[t];
            const float4 a1    = pk1_batch[t];
            const vec3 conic   = {a0.w, a1.x, a1.y};
            const vec3 xy_opac = {a0.x, a0.y, a0.z};
            const float opac   = xy_opac.z;
            float rgb_t[CDIM];
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgb_t[k] = k == 0 ? a1.z : (k == 1 ? a1.w : rgbs_batch[t * CDIM + k]);
            }
            bool valid[PPT];
            float alpha[PPT], vis[PPT];
            vec2 delta[PPT];
            bool any_valid = false;
#    pragma unroll
            for(uint32_t p = 0; p < PPT; ++p)
            {
                valid[p] = inside[p] && !(batch_end_offset - t > bin_final[p]);
                if(valid[p])
                {
                    delta[p]                = {xy_opac.x - px[p], xy_opac.y - py[p]};
                    const GaussianWeight gw = eval_gaussian_weight(conic, delta[p].x, delta[p].y, opac);
                    vis[p]                  = gw.vis;
                    alpha[p]                = gw.alpha;
                    valid[p]                = gw.valid;
                }
                any_valid |= valid[p];
            }
            if(!warp.any(any_valid))
            {
                continue;
            }
            float vals[16];
#    pragma unroll
            for(uint32_t k = 0; k < 16; ++k)
            {
                vals[k] = 0.f;
            }
#    pragma unroll
            for(uint32_t p = 0; p < PPT; ++p)
            {
                if(!valid[p])
                {
                    continue;
                }
                const float ra  = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha[p]);
                T[p]           *= ra;
                const float fac = alpha[p] * T[p];
                float v_alpha   = 0.f;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    vals[k] += fac * v_render_c[p][k];
                    v_alpha += (rgb_t[k] * T[p] - buffer[p][k] * ra) * v_render_c[p][k];
                }
                v_alpha += T_final[p] * ra * v_render_a[p];
                if(backgrounds != nullptr)
                {
                    v_alpha += -T_final[p] * ra * bg_dot[p];
                }
                if(opac * vis[p] <= MAX_ALPHA)
                {
                    const float v_sigma = -opac * vis[p] * v_alpha;
                    const float dx = delta[p].x, dy = delta[p].y;
                    vals[CDIM + 0] += 0.5f * v_sigma * dx * dx;
                    vals[CDIM + 1] += v_sigma * dx * dy;
                    vals[CDIM + 2] += 0.5f * v_sigma * dy * dy;
                    const float gx = v_sigma * (conic.x * dx + conic.y * dy);
                    const float gy = v_sigma * (conic.y * dx + conic.z * dy);
                    vals[CDIM + 3] += gx;
                    vals[CDIM + 4] += gy;
                    vals[CDIM + 5] += fabsf(gx);
                    vals[CDIM + 6] += fabsf(gy);
                    vals[CDIM + 7] += vis[p] * v_alpha;
                }
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[p][k] += rgb_t[k] * fac;
                }
            }
            const float sum  = warpReduceScatter16(vals, lane);
            const uint32_t c = lane >> 1;
            if((lane & 1) == 0 && c < CDIM + 8)
            {
                const int32_t g = id_batch[t];
                float *ptr      = nullptr;
                if(c < CDIM)
                    ptr = v_colors + CDIM * g + c;
                else if(c < CDIM + 3)
                    ptr = (float *)(v_conics) + 3 * g + (c - CDIM);
                else if(c < CDIM + 5)
                    ptr = (float *)(v_means2d) + 2 * g + (c - CDIM - 3);
                else if(c < CDIM + 7)
                    ptr = v_means2d_abs == nullptr ? nullptr : (float *)(v_means2d_abs) + 2 * g + (c - CDIM - 5);
                else
                    ptr = v_opacities + g;
                if(ptr != nullptr)
                    atomicAdd_system(ptr, sum);
            }
        }
    }
}

void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N           = packed ? 0 : means2d.size(-2);                        // number of gaussians
    uint32_t I           = render_alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads     = {tile_size, tile_size, 1};
    uint32_t n_tiles = I * tile_height * tile_width;
    dim3 grid        = {n_tiles, 1, 1};

    if(n_isects == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );

    auto launch_kernel = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

        if constexpr(CDIM + 8 <= 16)
        {
            if(tile_size == 16)
            {
                rasterize_to_pixels_3dgs_bwd_ppt_kernel<CDIM, BWD_PPT>
                    <<<n_tiles, 256 / BWD_PPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                        I,
                        n_isects,
                        reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                        reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                        colors.const_data_ptr<float>(),
                        opacities.const_data_ptr<float>(),
                        backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                        masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                        image_width,
                        image_height,
                        tile_width,
                        tile_height,
                        tile_offsets.const_data_ptr<int64_t>(),
                        flatten_ids.const_data_ptr<int32_t>(),
                        render_alphas.const_data_ptr<float>(),
                        last_ids.const_data_ptr<int32_t>(),
                        v_render_colors.const_data_ptr<float>(),
                        v_render_alphas.const_data_ptr<float>(),
                        v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                                  : nullptr,
                        reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                        reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                        v_colors.data_ptr<float>(),
                        v_opacities.data_ptr<float>()
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
        }

        int64_t shmem_size
            = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);

        if(cudaFuncSetAttribute(
               rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }

        rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                I,
                N,
                n_isects,
                packed,
                reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                colors.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                0,
                tile_offsets.const_data_ptr<int64_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
                render_alphas.const_data_ptr<float>(),
                last_ids.const_data_ptr<int32_t>(),
                v_render_colors.const_data_ptr<float>(),
                v_render_alphas.const_data_ptr<float>(),
                v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

void launch_rasterize_to_pixels_3dgs_bwd_kernels(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N           = packed ? 0 : means2d.size(-2);
    uint32_t I           = render_alphas.numel() / (image_height * image_width);
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);
    uint32_t n_tiles     = I * tile_height * tile_width;

    dim3 threads = {tile_size, tile_size, 1};

    if(n_isects == 0)
    {
        return;
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );

    auto launch_kernels = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

        int64_t shmem_size
            = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);

        for(const auto device_id: c10::irange(c10::cuda::device_count()))
        {
            C10_CUDA_CHECK(cudaSetDevice(device_id));
            if(cudaFuncSetAttribute(
                   rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   shmem_size
               )
               != cudaSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size."
                );
            }
            auto stream = c10::cuda::getCurrentCUDAStream(device_id);

            int64_t block_offset, block_count;
            std::tie(block_offset, block_count) = chunk(n_tiles, device_id);
            if(block_count > 0)
            {
                dim3 grid = {static_cast<uint32_t>(block_count), 1, 1};
                rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float><<<grid, threads, shmem_size, stream>>>(
                    I,
                    N,
                    n_isects,
                    packed,
                    reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                    reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                    masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                    image_width,
                    image_height,
                    tile_size,
                    tile_width,
                    tile_height,
                    block_offset,
                    tile_offsets.const_data_ptr<int64_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    render_alphas.const_data_ptr<float>(),
                    last_ids.const_data_ptr<int32_t>(),
                    v_render_colors.const_data_ptr<float>(),
                    v_render_alphas.const_data_ptr<float>(),
                    v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                              : nullptr,
                    reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                    reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>()
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernels));
    TORCH_CHECK(
        dispatched,
        "dispatch failed: no matching compile-time instantiation for runtime "
        "parameters"
    );

    merge_streams();
}
} // namespace gsplat

#endif
