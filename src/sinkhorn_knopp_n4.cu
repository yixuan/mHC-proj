#include <cmath>
#include <cuda_runtime.h>
#include "sinkhorn_knopp.cuh"

void sinkhorn_knopp_n4(
    const float* d_M, float* d_T, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
)
{
    mhc::sinkhorn_knopp_forward_batched(d_T, d_M, N, 4, num_iters, 1e-8, stream);
}

void sinkhorn_knopp_n4_backward(
    const float* d_G, const float* d_M, float* d_D, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
)
{
    mhc::sinkhorn_knopp_backward_batched(d_D, d_G, d_M, N, 4, num_iters, 1e-8, stream);
}
