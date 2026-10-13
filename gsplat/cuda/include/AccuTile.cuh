/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SNUGBOX + AccuTile tile intersection (moved from IntersectTile.cu so the
// fused projection can count tiles with exactly the same code).
#pragma once

#include <cstdint>

#include "Common.h"

namespace gsplat
{

// ============================================================
// SNUGBOX + AccuTile helper functions as propsed by SpeedySplat: https://arxiv.org/pdf/2412.00578
// ============================================================

inline __device__ float2
    accutile_ellipse_intersection(float A, float B, float C, float disc, float t, float2 p, bool isY, float coord)
{
    float p_u   = isY ? p.y : p.x;
    float p_v   = isY ? p.x : p.y;
    float coeff = isY ? A : C;

    float h         = coord - p_u;
    float sqrt_term = sqrtf(disc * h * h + t * coeff);

    return {(-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v};
}

inline __device__ uint32_t accutile_process_tiles(
    float A,
    float B,
    float C,
    float disc,
    float t,
    float2 p,
    float2 bbox_min,
    float2 bbox_max,
    float2 bbox_argmin,
    float2 bbox_argmax,
    int2 rect_min,
    int2 rect_max,
    uint32_t tile_size,
    uint32_t tile_width,
    uint32_t n_tiles,
    bool isY,
    int64_t image_id,
    int64_t key_start,
    int64_t key_end,
    int64_t iid_enc,
    uint32_t tile_n_bits,
    int64_t depth_id_enc,
    uint32_t flatten_idx,
    int64_t *isect_ids,
    int32_t *flatten_ids,
    int64_t *cur_idx
)
{
    float BLOCK = (float)tile_size;

    if(isY)
    {
        rect_min    = {rect_min.y, rect_min.x};
        rect_max    = {rect_max.y, rect_max.x};
        bbox_min    = {bbox_min.y, bbox_min.x};
        bbox_max    = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK;
    if(bbox_min.x <= min_line)
    {
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    }
    else
    {
        intersect_min_line = intersect_max_line;
    }

#pragma unroll 1
    for(int u = rect_min.x; u < rect_max.x; ++u)
    {
        max_line = min_line + BLOCK;
        if(max_line <= bbox_max.x)
        {
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
        }

        if(min_line <= bbox_argmin.y && bbox_argmin.y < max_line)
        {
            ellipse_min = bbox_min.y;
        }
        else
        {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if(min_line <= bbox_argmax.y && bbox_argmax.y < max_line)
        {
            ellipse_max = bbox_max.y;
        }
        else
        {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)));
        int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1)));

#pragma unroll 1
        for(int v = min_tile_v; v < max_tile_v; v++)
        {
            int64_t tile_id  = isY ? (int64_t)(u * tile_width + v) : (int64_t)(v * tile_width + u);
            int64_t tile_key = image_id * n_tiles + tile_id;
            if(tile_key < key_start || tile_key >= key_end)
            {
                continue;
            }
            ++tiles_count;

            if(isect_ids != nullptr)
            {
                isect_ids[*cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[*cur_idx] = static_cast<int32_t>(flatten_idx);
                ++(*cur_idx);
            }
        }

        intersect_min_line = intersect_max_line;
        min_line           = max_line;
    }
    return tiles_count;
}

// Per-gaussian setup of the AccuTile enumeration (the part of
// intersect_tile_kernel before accutile_process_tiles).
struct AccuTileSetup
{
    float disc, t;
    float2 bbox_min, bbox_max, bbox_argmin, bbox_argmax;
    int2 rect_min, rect_max;
    bool isY, empty;
};

inline __device__ AccuTileSetup accutile_setup(
    const float A, const float B, const float C, const float opacity, const float2 mean2d, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height
)
{
    AccuTileSetup su;
    // Quadratic form: a*dx^2 + 2*b*dx*dy + c*dy^2
    // disc = B^2 - A*C = -(det Sigma^{-1})
    su.disc = B * B - A * C;

    // Opacity-aware isocontour level: alpha = opacity * exp(-0.5 * q) >= ALPHA_THRESHOLD
    // => q <= 2 * ln(opacity / ALPHA_THRESHOLD). Cap at GAUSSIAN_EXTEND^2 (same as the gsplat radius budget).
    su.t = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity / ALPHA_THRESHOLD));

    // SNUGBOX: tight axis-aligned bounding box of the ellipse
    float neg_t_over_disc = -su.t / su.disc;
    float x_extent        = sqrtf(neg_t_over_disc * C);
    float y_extent        = sqrtf(neg_t_over_disc * A);

    su.bbox_min = {mean2d.x - x_extent, mean2d.y - y_extent};
    su.bbox_max = {mean2d.x + x_extent, mean2d.y + y_extent};

    float Bx_over_C = B * x_extent / C;
    float By_over_A = B * y_extent / A;
    su.bbox_argmin  = {mean2d.y + Bx_over_C, mean2d.x + By_over_A};
    su.bbox_argmax  = {mean2d.y - Bx_over_C, mean2d.x - By_over_A};

    float tile_size_f = (float)tile_size;
    su.rect_min
        = {max(0, min((int)tile_width, (int)(su.bbox_min.x / tile_size_f))),
           max(0, min((int)tile_height, (int)(su.bbox_min.y / tile_size_f)))};
    su.rect_max
        = {max(0, min((int)tile_width, (int)(su.bbox_max.x / tile_size_f + 1.f))),
           max(0, min((int)tile_height, (int)(su.bbox_max.y / tile_size_f + 1.f)))};

    int y_span = su.rect_max.y - su.rect_min.y;
    int x_span = su.rect_max.x - su.rect_min.x;
    su.empty   = y_span * x_span == 0;
    su.isY     = y_span < x_span;
    return su;
}

} // namespace gsplat
