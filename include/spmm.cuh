#ifndef SPMM_CUH
#define SPMM_CUH

#include "common.h"
#include "CSR.hpp"

namespace spmm {

__global__ void SpMM(const size_t m, const size_t n, const size_t k,
                     const float *d_A_vals,
                     const uint32_t *d_A_colinds,
                     const uint32_t *d_A_rowptrs,
                     const float *d_X,
                     float *d_Y)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31;              // thread's lane within warp
    const uint32_t warp_in_block = tid >> 5;     // warp index within block
    const uint32_t warps_per_block = 4;

    const uint32_t row = blockIdx.x * warps_per_block + warp_in_block;

    if (row >= m) return;

    const uint32_t row_start = d_A_rowptrs[row];
    const uint32_t row_end   = d_A_rowptrs[row + 1];

    // Each lane computes a subset of columns of Y(row, :)
    for (uint32_t j = lane; j < k; j += 32) {
        float sum = 0.0f;

        for (uint32_t idx = row_start; idx < row_end; idx++) {
            const uint32_t a_col = d_A_colinds[idx];
            const float a_val = d_A_vals[idx];

            // row-major X
            sum += a_val * d_X[a_col * k + j];
        }

        // row-major Y
        d_Y[row * k + j] = sum;
    }
}
void SpMM_wrapper(csr_t& A, float * d_X, float * d_Y, const size_t k)
{
    //**** CHANGE THESE VALUES ****//
    const uint32_t threads_per_block = 128; // 4 warps
    const uint32_t warps_per_block = threads_per_block / 32;

    const uint32_t blocks =
        (A.get_rows() + warps_per_block - 1) / warps_per_block;


    //**** OPTIONAL: Preprocessing/load balancing step ****//
    // TODO

    // Call the kernel
    SpMM<<<blocks, threads_per_block>>>(A.get_rows(), A.get_cols(), k,
                                        A.get_vals(), A.get_colinds(), A.get_rowptrs(), 
                                        d_X, d_Y);

    // Sync w/ the host
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
