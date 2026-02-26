// Input tensor is R [N x 4 x 4], so R[k, :, :] is the k-th matrix instance of size 4x4
// let each warp process two instances:
//   Lane 0-15 for instance k, lane 16-31 for instance k+1
// For the first instance thread group (16 threads) in the warp, map each thread to a matrix entry (row, col):
//   lid = 0, ..., 15
//   lid_gr = 0, ..., 15
//   row = lid_gr / 4
//   col = lid_gr % 4
//   warp mask = 0x0000ffff
// For the second instance thread group:
//   lid = 16, ..., 31
//   lid_gr = lid - 16 = 0, ..., 15
//   row = lid_gr / 4
//   col = lid_gr % 4
//   warp mask = 0xffff0000

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

//========================
// Macros
//========================

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

//========================
// Constants
//========================

#define BLOCK_DIM 256
#define THREADS_PER_INSTANCE 16
#define NEWTON_MAX_ITERS 20
#define LINE_SEARCH_MAX_ITERS 5
#define EPSILON 1e-8f

//========================
// Warp primitives
//========================

// Intra-warp row sum
//
// Original array:
//     t = [t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33]
// After calling val += __shfl_xor_sync(mask, val, 1):
//    t += [t01, t00, t03, t02, t11, t10, ...]                                0<->1, 2<->3, 4<->5, 6<->7, ...
// ==> t = [t00 + t01, t00 + t01, t02 + t03, t02 + t03, ...]
// After calling val += __shfl_xor_sync(mask, val, 2):
//    t += [t02 + t03, t02 + t03, t00 + t01, t00 + t01, ...]                  0<->2, 1<->3, 4<->6, 5<->7, ...
// ==> t = [t00 + t01 + t02 + t03, t00 + t01 + t02 + t03, t00 + t01 + t02 + t03, ...]
//
// The two instance groups (lane 0-15 and lane 16-31) do not interfere with each other
// with proper masks:
//     lane_id < 16 => 0x0000ffff, lane_id >= 16 => 0xffff0000
//
// For all threads with row == i, val contains the row sum value
__device__ __forceinline__ float warp_reduce_sum_row(float val, unsigned int mask)
{
    val += __shfl_xor_sync(mask, val, 1);
    val += __shfl_xor_sync(mask, val, 2);
    return val;
}

// Similar operation for computing the rowwise max value
__device__ __forceinline__ float warp_reduce_max_row(float val, unsigned int mask)
{
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
    return val;
}

// Intra-warp column sum
//
// Original array:
//     t = [t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33]
// After calling val += __shfl_xor_sync(0xffffffff, val, 4):
//    t += [t10, t11, t12, t13, t00, t01, t02, t03, ...]                      0<->4, 1<->5, 2<->6, 3<->7, ...
// ==> t = [t00 + t10, t01 + t11, t02 + t12, t03 + t13, ...]
// After calling val += __shfl_xor_sync(0xffffffff, val, 8):
//    t += [t20 + t30, t21 + t31, t22 + t32, t23 + t33, ...]                  0<->8, 1<->9, 2<->10, 3<->11, ...
// ==> t = [t00 + t10 + t20 + t30, t01 + t11 + t21 + t31, t02 + t12 + t22 + t32, ...]
//
// The two instance groups (lane 0-15 and lane 16-31) do not interfere with each other
// with proper masks:
//     lane_id < 16 => 0x0000ffff, lane_id >= 16 => 0xffff0000
//
// For all threads with col == j, val contains the column sum value
__device__ __forceinline__ float warp_reduce_sum_col(float val, unsigned int mask)
{
    val += __shfl_xor_sync(mask, val, 4);
    val += __shfl_xor_sync(mask, val, 8);
    return val;
}

// Similar operation for computing the columnwise max value
__device__ __forceinline__ float warp_reduce_max_col(float val, unsigned int mask)
{
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
    return val;
}

// Broadcast value at a specific column to the whole row
//
// (lane_id & ~3) sets the last two bits of lane_id to zero, which corresponds to
// the lane ID of matrix entry (row, 0), for both instance groups
// For src_col = 0, 1, 2, 3, ((lane_id & ~3) | src_col) corresponds to (row, src_col)
// So this function basically returns the value of (row, src_col) for the current thread at (row, j)
__device__ __forceinline__ float warp_broadcast_row(float val, int lane_id, int src_col, unsigned int mask)
{
    return __shfl_sync(mask, val, (lane_id & (~3)) | src_col);
}

//========================
// Device helper functions
//========================

// Initialize beta using one Sinkhorn iteration from alpha=0
// objfn = n - sum(alpha) - sum(beta)
// Return f = -sum(alpha) - sum(beta) at (alpha, beta) = (0, beta)
__device__ __forceinline__ float initialize_beta_alpha0
(
    float val_R, float& f_alpha0, unsigned int base_lane_id, unsigned int mask
)
{
    // beta[j] = -log(sum_i exp(alpha[i] + R[i, j]))
    // Set alpha[i] = 0, and then
    //   beta[j] = -log(sum_i exp(R[i, j]))
    //           = -nu[j] - log(sum_i exp(R[i, j] - nu[j]))
    //   nu[j] = max_i R[i, j]
    const float col_max = warp_reduce_max_col(val_R, mask);
    const float col_sum_exp = warp_reduce_sum_col(expf(val_R - col_max), mask);
    const float val_beta = -col_max - logf(col_sum_exp);
    // f = -sum(alpha) - sum(beta) = -sum(beta)
    f_alpha0 = -warp_reduce_sum_row(val_beta, mask);
    // Shift beta to make beta[m] = beta[3] = 0
    const float betam = __shfl_sync(mask, val_beta, base_lane_id + 3);
    return val_beta - betam;
}

// Update beta using one cycle of Sinkhorn iterations
__device__ __forceinline__ float sinkhorn_iteration
(
    float val_beta, float val_R, unsigned int base_lane_id, unsigned int mask
)
{
    // alpha[i] = -log(sum_j exp(beta[j] + R[i, j]))
    //          = -mu[i] - log(sum_j exp(U[i, j] - mu[i]))
    // U[i, j] = beta[j] + R[i, j]
    //   mu[i] = max_j U[i, j]
    const float val_u = val_beta + val_R;
    const float row_max = warp_reduce_max_row(val_u, mask);
    const float row_sum_exp = warp_reduce_sum_row(expf(val_u - row_max), mask);
    const float val_alpha = -row_max - logf(row_sum_exp);

    // beta[j] = -log(sum_i exp(alpha[i] + R[i, j]))
    //         = -nu[j] - log(sum_i exp(V[i, j] - nu[j]))
    // V[i, j] = alpha[i] + R[i, j]
    //   nu[j] = max_i V[i, j]
    const float val_v = val_alpha + val_R;
    const float col_max = warp_reduce_max_col(val_v, mask);
    const float col_sum_exp = warp_reduce_sum_col(expf(val_v - col_max), mask);
    val_beta = -col_max - logf(col_sum_exp);
    // Shift beta to make beta[m] = beta[3] = 0
    const float betam = __shfl_sync(mask, val_beta, base_lane_id + 3);
    return val_beta - betam;
}

// Compute the gradient value and gradient norm for current beta
//
// For thread at (row, col):
//     val_beta = { beta[col], col = 0, 1, 2
//                { 0.0，      col = 3
//     val_R = R[row, col]
//     out_c = c[col]
//     out_g = { g[col] = c[col] - 1, col = 0, 1, 2
//             { 0.0,                 col = 3
// Since g = c - 1, we only write out c
//
// base_lane_id is the first lane in the instance thread group,
// base_lane_id = { 0,  lane_id = 0, ..., 15
//                { 15, lane_id = 16, ..., 31
// base_lane_id corresponds to the (0, 0) entry of the matrix
__device__ __forceinline__ float compute_gradient
(
    float val_beta, float val_R,
    int row, int col, int base_lane_id, unsigned int mask,
    float& out_c, float& out_T
)
{
    // Compute T[row, col]
    const float val_u = val_beta + val_R;
    const float row_max = warp_reduce_max_row(val_u, mask);
    const float val_exp = expf(val_u - row_max);
    const float row_sum_exp = warp_reduce_sum_row(val_exp, mask);
    const float val_T = val_exp / row_sum_exp;
    out_T = val_T;

    // Compute c[col], c = T'1 = colsum(T)
    const float val_c = warp_reduce_sum_col(val_T, mask);
    out_c = val_c;

    // Compute gradient g[col], g = c - 1, g[3] = 0.0
    const float val_g = (col < 3) ? (val_c - 1.0f) : (0.0f);

    // Compute gradient norm, |g0| + |g1| + |g2| + |g3| = |g0| + |g1| + |g2|
    const float gnorm = warp_reduce_sum_row(fabsf(val_g), mask);

    // The whole instance thread group holds the same gnorm value
    return gnorm;
}

// Compute objective function value and gradient norm
__device__ __forceinline__ float compute_f_gradient
(
    float val_beta, float val_R,
    int row, int col, int base_lane_id, unsigned int mask,
    float& out_c, float& out_T, float& f
)
{
    // Compute T[row, col]
    const float val_u = val_beta + val_R;
    const float row_max = warp_reduce_max_row(val_u, mask);
    const float val_exp = expf(val_u - row_max);
    const float row_sum_exp = warp_reduce_sum_row(val_exp, mask);
    const float val_alpha = -row_max - logf(row_sum_exp);
    const float val_T = val_exp / row_sum_exp;
    out_T = val_T;

    // Compute c[col], c = T'1 = colsum(T)
    const float val_c = warp_reduce_sum_col(val_T, mask);
    out_c = val_c;

    // Compute (part of) objective function
    // objfn = n - sum(alpha) - sum(beta)
    // Return f = -sum(alpha) - sum(beta)
    f = -warp_reduce_sum_col(val_alpha, mask) - warp_reduce_sum_row(val_beta, mask);

    // Compute gradient g[col], g = c - 1, g[3] = 0.0
    const float val_g = (col < 3) ? (val_c - 1.0f) : (0.0f);

    // Compute gradient norm, |g0| + |g1| + |g2| + |g3| = |g0| + |g1| + |g2|
    const float gnorm = warp_reduce_sum_row(fabsf(val_g), mask);

    // The whole instance thread group holds the same gnorm value
    return gnorm;
}

// Compute the Newton direction for current beta
//
// For thread at (row, col):
//     val_c = c[col]
//     val_g = { g[col] = c[col] - 1, col = 0, 1, 2
//             { 0.0,                 col = 3
__device__ __forceinline__ float compute_newton_direction(
    float val_c, float val_T, float gnorm,
    int row, int col, int lane_id, int base_lane_id, unsigned int mask
)
{
    // Compute P[j, k] = sum_i (T[i, j] * T[i, k])

    /*
    // (t0, t1, t2) -> (T[row, 0], T[row, 1], T[row, 2])
    // Threads at the same row hold the same values of t0, t1, t2
    float t0 = warp_broadcast_row(val_T, lane_id, 0, mask);
    float t1 = warp_broadcast_row(val_T, lane_id, 1, mask);
    float t2 = warp_broadcast_row(val_T, lane_id, 2, mask);

    // Each p** variable has the same value across all threads
    float p00 = warp_reduce_sum_col(t0 * t0, mask);
    float p01 = warp_reduce_sum_col(t0 * t1, mask);
    float p02 = warp_reduce_sum_col(t0 * t2, mask);
    float p11 = warp_reduce_sum_col(t1 * t1, mask);
    float p12 = warp_reduce_sum_col(t1 * t2, mask);
    float p22 = warp_reduce_sum_col(t2 * t2, mask);
    */

    // Column sums of T .* T gives p00, p11, p22
    // hii = ci - pii
    // Add ||g||^2 to the diagonal to improve numerical stability
    const float diag = fminf(gnorm * gnorm, 1e-3f);
    const float hii = diag + val_c - warp_reduce_sum_col(val_T * val_T, mask);
    // Broadcast to all threads
    const float h00 = __shfl_sync(mask, hii, base_lane_id);
    const float h11 = __shfl_sync(mask, hii, base_lane_id + 1);
    const float h22 = __shfl_sync(mask, hii, base_lane_id + 2);

    // To get p01, p12, p02, we need to shuffle the columns of T
    // in the following way:
    //    T[i, 0], T[i, 1], T[i, 2], T[i, 3]
    // => T[i, 1], T[i, 2], T[i, 0], T[i, *]
    //
    // Use the width argument to auto-group
    const int src_col = (col >= 2) ? 0 : (col + 1);
    const float val_T_perm = __shfl_sync(mask, val_T, src_col, 4);
    // hij = -pij, for i != j
    const float hij = -warp_reduce_sum_col(val_T * val_T_perm, mask);
    // Broadcast to all threads
    const float h01 = __shfl_sync(mask, hij, base_lane_id);
    const float h12 = __shfl_sync(mask, hij, base_lane_id + 1);
    const float h02 = __shfl_sync(mask, hij, base_lane_id + 2);

    // g[col] = c[col] - 1, col = 0, 1, 2
    const float val_g = val_c - 1.0f;
    // Broadcast to all threads
    const float g0 = __shfl_sync(mask, val_g, base_lane_id);
    const float g1 = __shfl_sync(mask, val_g, base_lane_id + 1);
    const float g2 = __shfl_sync(mask, val_g, base_lane_id + 2);

    // det(H)
    // H is theoretically positive definite, but we need to be careful
    // on the numerical stability
    float det = h00 * (h11 * h22 - h12 * h12) -
                h01 * (h01 * h22 - h12 * h02) +
                h02 * (h01 * h12 - h11 * h02);
    // Let (lam1, lam2, lam3) be the three eigenvalues of H
    // lam1 + lam2 + lam3 = h00 + h11 + h22
    // lam1 * lam2 * lam3 = det
    // Define m = (h00 + h11 + h22) / 3 to be the arithmetic mean
    // det^(1/3) is the geometric mean
    // Then rho = m^3 / det >= 1 measures how disperse the three eigenvalues of H are
    // If rho is large, then the condition number of H may also be large
    // In this case, we switch to gradient descent as a fallback method by setting d = -g
    // Similarly, if det < EPSILON, then it means that the Newton direction may be very unstable,
    // so we switch to gradient descent
    const float m = (h00 + h11 + h22) / 3.0f;
    const float rho = m * m * m / det;
    if (det <= EPSILON || rho > 1000)
    {
        return (col < 3) ? -val_g : 0.0f;
    }

    // Compute d = -H^{-1}g
    float val_d = 0.0f;
    float val_y = 0.0f;

    // Only threads with col = 0, 1, 2 participate in solving 3x3 linear system
    if (col == 0)
    {
        val_y = (h11 * h22 - h12 * h12) * g0 +
                (h02 * h12 - h01 * h22) * g1 +
                (h01 * h12 - h11 * h02) * g2;
    }
    else if (col == 1)
    {
        val_y = (h02 * h12 - h01 * h22) * g0 +
                (h00 * h22 - h02 * h02) * g1 +
                (h01 * h02 - h00 * h12) * g2;
    }
    else if (col == 2) 
    {
        val_y = (h01 * h12 - h11 * h02) * g0 +
                (h01 * h02 - h00 * h12) * g1 +
                (h00 * h11 - h01 * h01) * g2;
    }
    val_d = -val_y / det;

    return val_d; 
}

//========================
// Forward pass kernel
//========================

// In: R [N x 4 x 4]
// Out: T [N x 4 x 4]
__global__ void birkhoff_proj_n4_kernel(
    const float* __restrict__ R,
    float* __restrict__ T,
    float tol,
    int N
)
{
    // Indices
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int instance_id = global_tid / THREADS_PER_INSTANCE;
    if (instance_id >= N)
    {
        return;
    }

    const int lane_id = threadIdx.x & 31;
    const int lane_id_gr = threadIdx.x % THREADS_PER_INSTANCE;
    const int base_lane_id = lane_id & (~15);
    const unsigned int active_mask = (lane_id < 16) ? 0x0000ffff : 0xffff0000;
    const int row = lane_id_gr / 4;
    const int col = lane_id_gr % 4;

    // Load R
    const int ind_mat = instance_id * 16 + lane_id_gr;
    const float val_R = R[ind_mat];
    
    // Initialize beta to be zero
    float val_beta = 0.0f;

    // Compute the objective function value and gradient
    float val_c = 0.0f, val_T = 0.0f, current_f = 0.0f;
    float current_gnorm = compute_f_gradient(
        val_beta, val_R, row, col, base_lane_id, active_mask,
        val_c, val_T, current_f
    );

    // Another reasonable initialization is to set alpha=0
    // We compare the two methods, beta=0 and alpha=0, and choose the one
    // with smaller objective function value
    float f_alpha0 = 0.0f;
    float val_beta_alpha0 = initialize_beta_alpha0(val_R, f_alpha0, base_lane_id, active_mask);
    if (f_alpha0 < current_f)
    {
        val_beta = val_beta_alpha0;
        current_gnorm = compute_f_gradient(
            val_beta, val_R, row, col, base_lane_id, active_mask,
            val_c, val_T, current_f
        );
    }

    // Test convergence
    if (current_gnorm < tol)
    {
        // Write T
        T[ind_mat] = val_T;
        return;
    }

    // Step sizes for line search
    constexpr float gamma_list[5] = {1.0f, 0.5f, 0.1f, 0.05f, 0.01f};

    // Newton iterations
    #pragma unroll 1
    for (int iter = 0; iter < NEWTON_MAX_ITERS; iter++)
    {
        // Newton direction
        // For thread (row, col) with col = 0, 1, 2, val_d = d[col]
        // val_d = 0 for col = 3
        const float val_d = compute_newton_direction(
            val_c, val_T, current_gnorm, row, col, lane_id, base_lane_id, active_mask
        );

        // Line search
        // bool step_accepted = false;
        #pragma unroll 1
        for (int k = 0; k < LINE_SEARCH_MAX_ITERS; k++)
        {
            const float gamma = gamma_list[k];

            // Candidate beta value
            // Note that val_beta = 0 and val_d = 0 for col = 3, so we also have
            // candidate_beta = 0 for col = 3
            const float candidate_beta = val_beta + gamma * val_d;

            // Compute gradient
            float candidate_c = 0.0f, candidate_T = 0.0f, candidate_f = 0.0f;
            const float candidate_gnorm = compute_f_gradient(
                candidate_beta, val_R, row, col, base_lane_id, active_mask,
                candidate_c, candidate_T, candidate_f
            );

            // Test line search condition (objective function value and gradient norm decrease)
            if (candidate_f < current_f && candidate_gnorm < current_gnorm)
            {
                // Accept step and update related variables
                val_beta = candidate_beta;
                val_c = candidate_c;
                val_T = candidate_T;
                current_f = candidate_f;
                current_gnorm = candidate_gnorm;
                // step_accepted = true;
                break;
            }

            // Fallback to Sinkhorn iteration
            if (k == (LINE_SEARCH_MAX_ITERS - 1))
            {
                val_beta = sinkhorn_iteration(val_beta, val_R, base_lane_id, active_mask);
                current_gnorm = compute_f_gradient(
                    val_beta, val_R, row, col, base_lane_id, active_mask,
                    val_c, val_T, current_f
                );
            }
        }

        // Test convergence
        if (current_gnorm < tol)
        {
            break;
        }
    }

    // Write T
    T[ind_mat] = val_T;
}

//========================
// Launcher
//========================

void birkhoff_proj_n4(
    const float* d_R, float* d_T, float tol, int N, cudaStream_t stream = cudaStreamPerThread
)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (N * 16 + threadsPerBlock.x - 1) / threadsPerBlock.x; 

    birkhoff_proj_n4_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_R, d_T, tol, N
    );
}

//========================
// Backward pass kernel
//========================

// Compute x = Delta^{-1} * rhs
// Delta is the Hessian matrix computed from T, rhs is a length-3 vector
// For thread (row, col), val_T = T[row, col], val_rhs = rhs[col]
// val_rhs is not referenced for col = 3
// For thread (row, col) with col = 0, 1, 2, returned val_x = x[col]
// val_x = 0 for col = 3
__device__ __forceinline__ float solve_delta_linear_system(
    float val_T, float val_rhs,
    int row, int col, int lane_id, int base_lane_id, unsigned int mask
)
{
    // Compute P[j, k] = sum_i (T[i, j] * T[i, k])

    /*
    // (t0, t1, t2) -> (T[row, 0], T[row, 1], T[row, 2])
    // Threads at the same row hold the same values of t0, t1, t2
    float t0 = warp_broadcast_row(val_T, lane_id, 0, mask);
    float t1 = warp_broadcast_row(val_T, lane_id, 1, mask);
    float t2 = warp_broadcast_row(val_T, lane_id, 2, mask);

    // Each h** variable has the same value across all threads
    float h00 = 1.0f - warp_reduce_sum_col(t0 * t0, mask);
    float h01 = -warp_reduce_sum_col(t0 * t1, mask);
    float h02 = -warp_reduce_sum_col(t0 * t2, mask);
    float h11 = 1.0f - warp_reduce_sum_col(t1 * t1, mask);
    float h12 = -warp_reduce_sum_col(t1 * t2, mask);
    float h22 = 1.0f - warp_reduce_sum_col(t2 * t2, mask);
    */

    // Column sums of T .* T gives p00, p11, p22
    // hii = ci - pii = 1 - pii
    // We recompute c = T'1 = colsum(T) for better numerical stability
    const float val_c = warp_reduce_sum_col(val_T, mask);
    const float hii = val_c - warp_reduce_sum_col(val_T * val_T, mask);
    // Broadcast to all threads
    const float h00 = __shfl_sync(mask, hii, base_lane_id);
    const float h11 = __shfl_sync(mask, hii, base_lane_id + 1);
    const float h22 = __shfl_sync(mask, hii, base_lane_id + 2);

    // To get p01, p12, p02, we need to shuffle the columns of T
    // in the following way:
    //    T[i, 0], T[i, 1], T[i, 2], T[i, 3]
    // => T[i, 1], T[i, 2], T[i, 0], T[i, *]
    //
    // Use the width argument to auto-group
    const int src_col = (col >= 2) ? 0 : (col + 1);
    const float val_T_perm = __shfl_sync(mask, val_T, src_col, 4);
    // hij = -pij, for i != j
    const float hij = -warp_reduce_sum_col(val_T * val_T_perm, mask);
    // Broadcast to all threads
    const float h01 = __shfl_sync(mask, hij, base_lane_id);
    const float h12 = __shfl_sync(mask, hij, base_lane_id + 1);
    const float h02 = __shfl_sync(mask, hij, base_lane_id + 2);

    // det(H)
    // H is theoretically positive definite
    float det = h00 * (h11 * h22 - h12 * h12) -
                h01 * (h01 * h22 - h12 * h02) +
                h02 * (h01 * h12 - h11 * h02);
    det = fmaxf(det, EPSILON);

    // rhs values are stored in the first three columns,
    // so we need to broadcast
    const float rhs0 = __shfl_sync(mask, val_rhs, base_lane_id);
    const float rhs1 = __shfl_sync(mask, val_rhs, base_lane_id + 1);
    const float rhs2 = __shfl_sync(mask, val_rhs, base_lane_id + 2);

    // Compute x = Delta^{-1} rhs
    float val_x = 0.0f;
    float val_y = 0.0f;

    // Only threads with col = 0, 1, 2 participate in solving 3x3 linear system
    if (col == 0)
    {
        val_y = (h11 * h22 - h12 * h12) * rhs0 +
                (h02 * h12 - h01 * h22) * rhs1 +
                (h01 * h12 - h11 * h02) * rhs2;
    }
    else if (col == 1)
    {
        val_y = (h02 * h12 - h01 * h22) * rhs0 +
                (h00 * h22 - h02 * h02) * rhs1 +
                (h01 * h02 - h00 * h12) * rhs2;
    }
    else if (col == 2) 
    {
        val_y = (h01 * h12 - h11 * h02) * rhs0 +
                (h01 * h02 - h00 * h12) * rhs1 +
                (h00 * h11 - h01 * h01) * rhs2;
    }
    val_x = val_y / det;

    return val_x; 
}

// In: G [N x 4 x 4]
// In: T [N x 4 x 4]
// Out: D [N x 4 x 4]
__global__ void birkhoff_proj_n4_backward_kernel(
    const float* __restrict__ G,
    const float* __restrict__ T,
    float* __restrict__ D,
    int N
)
{
    // Indices
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int instance_id = global_tid / THREADS_PER_INSTANCE;
    if (instance_id >= N)
    {
        return;
    }

    const int lane_id = threadIdx.x & 31;
    const int lane_id_gr = threadIdx.x % THREADS_PER_INSTANCE;
    const int base_lane_id = lane_id & (~15);
    const unsigned int active_mask = (lane_id < 16) ? 0x0000ffff : 0xffff0000;
    const int row = lane_id_gr / 4;
    const int col = lane_id_gr % 4;

    // Load G and T
    const int ind_mat = instance_id * 16 + lane_id_gr;
    const float val_G = G[ind_mat];
    const float val_T = T[ind_mat];

    // Gamma = G .* T
    const float val_Gamma = val_G * val_T;
    // Compute muc[col], muc = Gamma' * 1 = colsum(Gamma)
    const float val_muc = warp_reduce_sum_col(val_Gamma, active_mask);
    // Compute mur[row], mur = Gamma * 1 = rowsum(Gamma)
    const float val_mur = warp_reduce_sum_row(val_Gamma, active_mask);
    // Compute T' * mur
    // For every thread processing T[row, col], val_mur contains mur[row],
    // so val_T * val_mur is T[row, col] * mur[row]. Then we get T' * mur
    // by computing the column sums of T[row, col] * mur[row]
    const float val_Tmur = warp_reduce_sum_col(val_T * val_mur, active_mask);
    const float val_w_rhs = val_muc - val_Tmur;
    // Solve the linear system w = inv(Delta) * w_rhs
    // For thread (row, col) with col = 0, 1, 2, val_w = w[col]
    // val_w = 0 for col = 3
    const float val_w = solve_delta_linear_system(val_T, val_w_rhs, row, col, lane_id, base_lane_id, active_mask);
    // Compute v = mur - T * w
    const float val_v = val_mur - warp_reduce_sum_row(val_T * val_w, active_mask);
    // Compute D, D[row, col] = (v[row] + w[col] - G[row, col]) * T[row, col]
    const float val_D = (val_v + val_w - val_G) * val_T;

    // Write D - we need to flip its sign since R = -M
    D[ind_mat] = -val_D;
}

//========================
// Launcher
//========================

void birkhoff_proj_n4_backward(
    const float* d_G, const float* d_T, float* d_D, int N, cudaStream_t stream = cudaStreamPerThread
)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (N * 16 + threadsPerBlock.x - 1) / threadsPerBlock.x; 

    birkhoff_proj_n4_backward_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_G, d_T, d_D, N
    );
}
