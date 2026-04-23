#ifndef SPMM_CUH
#define SPMM_CUH

#include "common.h"
#include "CSR.hpp"

namespace spmm {

// ----------------------------------------------------------------------------
// Warp-per-row SpMM: Y = A * X, A in CSR, X/Y dense row-major.
//
// Key ideas vs. the starter kernel:
//
// 1. Cooperative A streaming. The starter has every lane re-read the same
//    (value, colind) pair for each output column j. With 32 lanes per warp
//    that is 32x redundant traffic on A. Here the 32 lanes of a warp issue
//    one coalesced load that produces 32 (value, colind) pairs at a time
//    and then broadcast them one by one via __shfl_sync. A is streamed once
//    per row.
//
// 2. Register-resident Y. Each lane owns a fixed slice of columns of the
//    output row and keeps its partial sums in registers across the entire
//    nnz walk, instead of recomputing a single output element at a time.
//
// 3. Vectorized X/Y. For k = 64 each lane owns a float2 (cols 2l, 2l+1) and
//    for k = 256 each lane owns two float4s (cols {4l..4l+3, 4l+128..4l+131}).
//    X loads and Y stores become 8- or 16-byte-wide transactions, cutting
//    the LDG/STG instruction count 2-4x versus scalar accesses. The lane
//    layout is still fully coalesced: the 32 lanes of a warp cover a
//    contiguous 256 B / 512 B chunk of X per transaction.
//
// 4. __restrict__ and __ldg. A and X are read-only; __ldg routes them
//    through the read-only / texture cache path, which helps X reuse.
// ----------------------------------------------------------------------------

template<uint32_t WARPS_PER_BLOCK>
__global__ void SpMM_k64_kernel(
    const uint32_t m,
    const float    * __restrict__ d_A_vals,
    const uint32_t * __restrict__ d_A_colinds,
    const uint32_t * __restrict__ d_A_rowptrs,
    const float    * __restrict__ d_X,
    float          * __restrict__ d_Y)
{
    constexpr uint32_t K = 64;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= m) return;

    const uint32_t row_start = d_A_rowptrs[row];
    const uint32_t row_end   = d_A_rowptrs[row + 1];

    float2 sum = make_float2(0.0f, 0.0f);

    const uint32_t nnz      = row_end - row_start;
    const uint32_t full_end = row_start + (nnz & ~31u);

    // --- Full chunks of 32 nonzeros: warp-wide cooperative load of A. ---
    for (uint32_t base = row_start; base < full_end; base += 32) {
        const uint32_t idx   = base + lane;
        const float    a_val = __ldg(d_A_vals    + idx);
        const uint32_t a_col = __ldg(d_A_colinds + idx);

        #pragma unroll
        for (int t = 0; t < 32; ++t) {
            const float    av = __shfl_sync(0xffffffffu, a_val, t);
            const uint32_t ac = __shfl_sync(0xffffffffu, a_col, t);
            const float2 x = __ldg(reinterpret_cast<const float2*>(
                d_X + (size_t)ac * K + 2u * lane));
            sum.x += av * x.x;
            sum.y += av * x.y;
        }
    }

    // --- Tail: fewer than 32 remaining nonzeros. ---
    const uint32_t tail = row_end - full_end;
    if (tail) {
        const uint32_t idx = full_end + lane;
        float    a_val = 0.0f;
        uint32_t a_col = 0u;
        if (idx < row_end) {
            a_val = __ldg(d_A_vals    + idx);
            a_col = __ldg(d_A_colinds + idx);
        }
        for (uint32_t t = 0; t < tail; ++t) {
            const float    av = __shfl_sync(0xffffffffu, a_val, t);
            const uint32_t ac = __shfl_sync(0xffffffffu, a_col, t);
            const float2 x = __ldg(reinterpret_cast<const float2*>(
                d_X + (size_t)ac * K + 2u * lane));
            sum.x += av * x.x;
            sum.y += av * x.y;
        }
    }

    *reinterpret_cast<float2*>(d_Y + (size_t)row * K + 2u * lane) = sum;
}

template<uint32_t WARPS_PER_BLOCK>
__global__ void SpMM_k256_kernel(
    const uint32_t m,
    const float    * __restrict__ d_A_vals,
    const uint32_t * __restrict__ d_A_colinds,
    const uint32_t * __restrict__ d_A_rowptrs,
    const float    * __restrict__ d_X,
    float          * __restrict__ d_Y)
{
    constexpr uint32_t K            = 256;
    constexpr uint32_t K_VEC_CHUNKS = K / (32u * 4u); // = 2 float4s per lane
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= m) return;

    const uint32_t row_start = d_A_rowptrs[row];
    const uint32_t row_end   = d_A_rowptrs[row + 1];

    float4 sums[K_VEC_CHUNKS];
    #pragma unroll
    for (uint32_t c = 0; c < K_VEC_CHUNKS; ++c)
        sums[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const uint32_t nnz      = row_end - row_start;
    const uint32_t full_end = row_start + (nnz & ~31u);

    for (uint32_t base = row_start; base < full_end; base += 32) {
        const uint32_t idx   = base + lane;
        const float    a_val = __ldg(d_A_vals    + idx);
        const uint32_t a_col = __ldg(d_A_colinds + idx);

        #pragma unroll
        for (int t = 0; t < 32; ++t) {
            const float    av    = __shfl_sync(0xffffffffu, a_val, t);
            const uint32_t ac    = __shfl_sync(0xffffffffu, a_col, t);
            const float*   x_row = d_X + (size_t)ac * K;
            #pragma unroll
            for (uint32_t c = 0; c < K_VEC_CHUNKS; ++c) {
                const float4 x = __ldg(reinterpret_cast<const float4*>(
                    x_row + c * 128u + 4u * lane));
                sums[c].x += av * x.x;
                sums[c].y += av * x.y;
                sums[c].z += av * x.z;
                sums[c].w += av * x.w;
            }
        }
    }

    const uint32_t tail = row_end - full_end;
    if (tail) {
        const uint32_t idx = full_end + lane;
        float    a_val = 0.0f;
        uint32_t a_col = 0u;
        if (idx < row_end) {
            a_val = __ldg(d_A_vals    + idx);
            a_col = __ldg(d_A_colinds + idx);
        }
        for (uint32_t t = 0; t < tail; ++t) {
            const float    av    = __shfl_sync(0xffffffffu, a_val, t);
            const uint32_t ac    = __shfl_sync(0xffffffffu, a_col, t);
            const float*   x_row = d_X + (size_t)ac * K;
            #pragma unroll
            for (uint32_t c = 0; c < K_VEC_CHUNKS; ++c) {
                const float4 x = __ldg(reinterpret_cast<const float4*>(
                    x_row + c * 128u + 4u * lane));
                sums[c].x += av * x.x;
                sums[c].y += av * x.y;
                sums[c].z += av * x.z;
                sums[c].w += av * x.w;
            }
        }
    }

    #pragma unroll
    for (uint32_t c = 0; c < K_VEC_CHUNKS; ++c)
        *reinterpret_cast<float4*>(d_Y + (size_t)row * K + c * 128u + 4u * lane) = sums[c];
}

// Generic fallback. Correct for any k; used only when k is not 64 or 256
// (e.g. for the tiny n16.mtx debug matrix if run with an unusual k).
__global__ void SpMM_generic_kernel(
    const size_t m, const size_t k,
    const float    * __restrict__ d_A_vals,
    const uint32_t * __restrict__ d_A_colinds,
    const uint32_t * __restrict__ d_A_rowptrs,
    const float    * __restrict__ d_X,
    float          * __restrict__ d_Y)
{
    const uint32_t lane            = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5;
    const uint32_t row             = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if ((size_t)row >= m) return;

    const uint32_t row_start = d_A_rowptrs[row];
    const uint32_t row_end   = d_A_rowptrs[row + 1];

    for (size_t j = lane; j < k; j += 32) {
        float s = 0.0f;
        for (uint32_t idx = row_start; idx < row_end; ++idx) {
            const uint32_t ac = d_A_colinds[idx];
            const float    av = d_A_vals   [idx];
            s += av * d_X[(size_t)ac * k + j];
        }
        d_Y[(size_t)row * k + j] = s;
    }
}

void SpMM_wrapper(csr_t& A, float * d_X, float * d_Y, const size_t k)
{
    const size_t   m_sz              = A.get_rows();
    const uint32_t m                 = static_cast<uint32_t>(m_sz);
    constexpr uint32_t WARPS_PER_BLOCK   = 4;
    constexpr uint32_t THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    const uint32_t blocks = static_cast<uint32_t>(
        (m_sz + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    if (k == 64) {
        SpMM_k64_kernel<WARPS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK>>>(
            m,
            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
            d_X, d_Y);
    } else if (k == 256) {
        SpMM_k256_kernel<WARPS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK>>>(
            m,
            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
            d_X, d_Y);
    } else {
        SpMM_generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            m_sz, k,
            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
            d_X, d_Y);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace spmm

#endif
