import torch
import tilelang
import tilelang.language as T


_N4 = 4
_EPS = 1e-8
_THREADS_PER_BLOCK = 256
_THREADS_PER_INSTANCE = 16
_NEWTON_MAX_ITERS = 20
_LINE_SEARCH_MAX_ITERS = 5


def _check_n4_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3 or tensor.shape[-2:] != (_N4, _N4):
        raise ValueError(f"{name} must be a tensor of size B x 4 x 4")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _float32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float32).contiguous()


@T.macro
def _warp_reduce_sum_row(val, mask):
    acc = T.alloc_var(T.float32, val)
    acc = acc + T.shfl_xor(acc, 1, mask=mask)
    acc = acc + T.shfl_xor(acc, 2, mask=mask)
    return acc


@T.macro
def _warp_reduce_max_row(val, mask):
    acc = T.alloc_var(T.float32, val)
    acc = T.max(acc, T.shfl_xor(acc, 1, mask=mask))
    acc = T.max(acc, T.shfl_xor(acc, 2, mask=mask))
    return acc


@T.macro
def _warp_reduce_sum_col(val, mask):
    acc = T.alloc_var(T.float32, val)
    acc = acc + T.shfl_xor(acc, 4, mask=mask)
    acc = acc + T.shfl_xor(acc, 8, mask=mask)
    return acc


@T.macro
def _warp_reduce_max_col(val, mask):
    acc = T.alloc_var(T.float32, val)
    acc = T.max(acc, T.shfl_xor(acc, 4, mask=mask))
    acc = T.max(acc, T.shfl_xor(acc, 8, mask=mask))
    return acc


@T.macro
def _abs_tir(x):
    return T.if_then_else(x < 0.0, -x, x)


@T.macro
def _compute_f_gradient(val_beta, val_R, col, mask):
    val_u = val_beta + val_R
    row_max = _warp_reduce_max_row(val_u, mask)
    val_exp = T.exp(val_u - row_max)
    row_sum_exp = _warp_reduce_sum_row(val_exp, mask)
    val_alpha = -row_max - T.log(row_sum_exp)
    val_T = val_exp / row_sum_exp

    val_c = _warp_reduce_sum_col(val_T, mask)
    f = -_warp_reduce_sum_col(val_alpha, mask) - _warp_reduce_sum_row(val_beta, mask)

    val_g = T.if_then_else(col < 3, val_c - 1.0, 0.0)
    gnorm = _warp_reduce_sum_row(_abs_tir(val_g), mask)
    return val_c, val_T, f, gnorm


@T.macro
def _initialize_beta_alpha0(val_R, base_lane_id, mask):
    col_max = _warp_reduce_max_col(val_R, mask)
    col_sum_exp = _warp_reduce_sum_col(T.exp(val_R - col_max), mask)
    val_beta = -col_max - T.log(col_sum_exp)
    f_alpha0 = -_warp_reduce_sum_row(val_beta, mask)
    beta_last = T.shfl_sync(val_beta, base_lane_id + 3, mask=mask)
    return val_beta - beta_last, f_alpha0


@T.macro
def _sinkhorn_iteration(val_beta, val_R, base_lane_id, mask):
    val_u = val_beta + val_R
    row_max = _warp_reduce_max_row(val_u, mask)
    row_sum_exp = _warp_reduce_sum_row(T.exp(val_u - row_max), mask)
    val_alpha = -row_max - T.log(row_sum_exp)

    val_v = val_alpha + val_R
    col_max = _warp_reduce_max_col(val_v, mask)
    col_sum_exp = _warp_reduce_sum_col(T.exp(val_v - col_max), mask)
    next_beta = -col_max - T.log(col_sum_exp)
    beta_last = T.shfl_sync(next_beta, base_lane_id + 3, mask=mask)
    return next_beta - beta_last


@T.macro
def _select_col3_value(col, y0, y1, y2, zero):
    return T.if_then_else(
        col == 0,
        y0,
        T.if_then_else(col == 1, y1, T.if_then_else(col == 2, y2, zero)),
    )


@T.macro
def _compute_newton_direction(val_c, val_T, gnorm, col, base_lane_id, mask):
    diag = T.min(gnorm * gnorm, 1e-3)
    hii = diag + val_c - _warp_reduce_sum_col(val_T * val_T, mask)
    h00 = T.shfl_sync(hii, base_lane_id, mask=mask)
    h11 = T.shfl_sync(hii, base_lane_id + 1, mask=mask)
    h22 = T.shfl_sync(hii, base_lane_id + 2, mask=mask)

    # width=4 makes src_col local to each 4-lane row subgroup.
    src_col = T.if_then_else(col >= 2, 0, col + 1)
    val_T_perm = T.shfl_sync(val_T, src_col, 4, mask=mask)
    hij = -_warp_reduce_sum_col(val_T * val_T_perm, mask)
    h01 = T.shfl_sync(hij, base_lane_id, mask=mask)
    h12 = T.shfl_sync(hij, base_lane_id + 1, mask=mask)
    h02 = T.shfl_sync(hij, base_lane_id + 2, mask=mask)

    val_g = T.if_then_else(col < 3, val_c - 1.0, 0.0)
    g0 = T.shfl_sync(val_g, base_lane_id, mask=mask)
    g1 = T.shfl_sync(val_g, base_lane_id + 1, mask=mask)
    g2 = T.shfl_sync(val_g, base_lane_id + 2, mask=mask)

    det = (
        h00 * (h11 * h22 - h12 * h12)
        - h01 * (h01 * h22 - h12 * h02)
        + h02 * (h01 * h12 - h11 * h02)
    )
    mean_diag = (h00 + h11 + h22) / 3.0
    rho = mean_diag * mean_diag * mean_diag / det

    y0 = (
        (h11 * h22 - h12 * h12) * g0
        + (h02 * h12 - h01 * h22) * g1
        + (h01 * h12 - h11 * h02) * g2
    )
    y1 = (
        (h02 * h12 - h01 * h22) * g0
        + (h00 * h22 - h02 * h02) * g1
        + (h01 * h02 - h00 * h12) * g2
    )
    y2 = (
        (h01 * h12 - h11 * h02) * g0
        + (h01 * h02 - h00 * h12) * g1
        + (h00 * h11 - h01 * h01) * g2
    )
    zero = val_T * 0.0
    val_y = _select_col3_value(col, y0, y1, y2, zero)

    val_d = -val_y / det
    fallback_d = T.if_then_else(col < 3, -val_g, 0.0)
    return T.if_then_else(T.Or(det <= _EPS, rho > 1000.0), fallback_d, val_d)


@T.macro
def _line_search_gamma(k):
    return T.if_then_else(
        k == 0,
        1.0,
        T.if_then_else(
            k == 1,
            0.5,
            T.if_then_else(k == 2, 0.1, T.if_then_else(k == 3, 0.05, 0.01)),
        ),
    )


@T.macro
def _solve_delta_linear_system(val_T, val_rhs, col, base_lane_id, mask):
    val_c = _warp_reduce_sum_col(val_T, mask)
    hii = val_c - _warp_reduce_sum_col(val_T * val_T, mask)
    h00 = T.shfl_sync(hii, base_lane_id, mask=mask)
    h11 = T.shfl_sync(hii, base_lane_id + 1, mask=mask)
    h22 = T.shfl_sync(hii, base_lane_id + 2, mask=mask)

    src_col = T.if_then_else(col >= 2, 0, col + 1)
    val_T_perm = T.shfl_sync(val_T, src_col, 4, mask=mask)
    hij = -_warp_reduce_sum_col(val_T * val_T_perm, mask)
    h01 = T.shfl_sync(hij, base_lane_id, mask=mask)
    h12 = T.shfl_sync(hij, base_lane_id + 1, mask=mask)
    h02 = T.shfl_sync(hij, base_lane_id + 2, mask=mask)

    det = (
        h00 * (h11 * h22 - h12 * h12)
        - h01 * (h01 * h22 - h12 * h02)
        + h02 * (h01 * h12 - h11 * h02)
    )
    det = T.max(det, _EPS)

    rhs0 = T.shfl_sync(val_rhs, base_lane_id, mask=mask)
    rhs1 = T.shfl_sync(val_rhs, base_lane_id + 1, mask=mask)
    rhs2 = T.shfl_sync(val_rhs, base_lane_id + 2, mask=mask)

    y0 = (
        (h11 * h22 - h12 * h12) * rhs0
        + (h02 * h12 - h01 * h22) * rhs1
        + (h01 * h12 - h11 * h02) * rhs2
    )
    y1 = (
        (h02 * h12 - h01 * h22) * rhs0
        + (h00 * h22 - h02 * h02) * rhs1
        + (h01 * h02 - h00 * h12) * rhs2
    )
    y2 = (
        (h01 * h12 - h11 * h02) * rhs0
        + (h01 * h02 - h00 * h12) * rhs1
        + (h00 * h11 - h01 * h01) * rhs2
    )
    zero = val_T * 0.0
    val_y = _select_col3_value(col, y0, y1, y2, zero)
    return val_y / det


@tilelang.jit
def _birkhoff_proj_n4_forward_kernel(R, T_out, tol):
    N = T.dynamic("N")
    dtype = T.float32

    R: T.Tensor((N, _N4, _N4), dtype)  # type: ignore
    T_out: T.Tensor((N, _N4, _N4), dtype)  # type: ignore

    num_blocks = T.ceildiv(N * _THREADS_PER_INSTANCE, _THREADS_PER_BLOCK)

    with T.Kernel(num_blocks, threads=_THREADS_PER_BLOCK) as (bx,):
        for tx in T.Parallel(_THREADS_PER_BLOCK):
            global_tid = bx * _THREADS_PER_BLOCK + tx
            instance_id = global_tid // _THREADS_PER_INSTANCE

            if instance_id < N:
                lane_id = tx & 31
                lane_id_group = tx % _THREADS_PER_INSTANCE
                base_lane_id = lane_id & (~15)
                active_mask = T.if_then_else(
                    lane_id < 16, T.uint32(0x0000FFFF), T.uint32(0xFFFF0000)
                )

                row = lane_id_group // 4
                col = lane_id_group % 4

                val_R = R[instance_id, row, col]
                val_beta = T.alloc_var(T.float32, val_R * 0.0)
                init_c, init_T, init_f, init_gnorm = _compute_f_gradient(
                    val_beta, val_R, col, active_mask
                )
                val_c = T.alloc_var(T.float32, init_c)
                val_T = T.alloc_var(T.float32, init_T)
                current_f = T.alloc_var(T.float32, init_f)
                current_gnorm = T.alloc_var(T.float32, init_gnorm)

                val_beta_alpha0, f_alpha0 = _initialize_beta_alpha0(
                    val_R, base_lane_id, active_mask
                )
                if f_alpha0 < current_f:
                    val_beta = val_beta_alpha0
                    val_c, val_T, current_f, current_gnorm = _compute_f_gradient(
                        val_beta, val_R, col, active_mask
                    )

                if current_gnorm >= tol:
                    for _ in T.serial(_NEWTON_MAX_ITERS):
                        val_d = _compute_newton_direction(
                            val_c, val_T, current_gnorm, col, base_lane_id, active_mask
                        )

                        for k in T.serial(_LINE_SEARCH_MAX_ITERS):
                            gamma = _line_search_gamma(k)
                            candidate_beta = val_beta + gamma * val_d
                            candidate_c, candidate_T, candidate_f, candidate_gnorm = (
                                _compute_f_gradient(
                                    candidate_beta, val_R, col, active_mask
                                )
                            )

                            if T.And(
                                candidate_f < current_f,
                                candidate_gnorm < current_gnorm,
                            ):
                                val_beta = candidate_beta
                                val_c = candidate_c
                                val_T = candidate_T
                                current_f = candidate_f
                                current_gnorm = candidate_gnorm
                                T.loop_break()

                            if k == (_LINE_SEARCH_MAX_ITERS - 1):
                                val_beta = _sinkhorn_iteration(
                                    val_beta, val_R, base_lane_id, active_mask
                                )
                                val_c, val_T, current_f, current_gnorm = (
                                    _compute_f_gradient(
                                        val_beta, val_R, col, active_mask
                                    )
                                )

                        if current_gnorm < tol:
                            T.loop_break()

                T_out[instance_id, row, col] = val_T


@tilelang.jit
def _birkhoff_proj_n4_backward_kernel(G, T_out, D):
    N = T.dynamic("N")
    dtype = T.float32

    G: T.Tensor((N, _N4, _N4), dtype)  # type: ignore
    T_out: T.Tensor((N, _N4, _N4), dtype)  # type: ignore
    D: T.Tensor((N, _N4, _N4), dtype)  # type: ignore

    num_blocks = T.ceildiv(N * _THREADS_PER_INSTANCE, _THREADS_PER_BLOCK)

    with T.Kernel(num_blocks, threads=_THREADS_PER_BLOCK) as (bx,):
        for tx in T.Parallel(_THREADS_PER_BLOCK):
            global_tid = bx * _THREADS_PER_BLOCK + tx
            instance_id = global_tid // _THREADS_PER_INSTANCE

            if instance_id < N:
                lane_id = tx & 31
                lane_id_group = tx % _THREADS_PER_INSTANCE
                base_lane_id = lane_id & (~15)
                active_mask = T.if_then_else(
                    lane_id < 16, T.uint32(0x0000FFFF), T.uint32(0xFFFF0000)
                )

                row = lane_id_group // 4
                col = lane_id_group % 4

                val_G = G[instance_id, row, col]
                val_T = T_out[instance_id, row, col]
                val_Gamma = val_G * val_T
                val_muc = _warp_reduce_sum_col(val_Gamma, active_mask)
                val_mur = _warp_reduce_sum_row(val_Gamma, active_mask)
                val_Tmur = _warp_reduce_sum_col(val_T * val_mur, active_mask)
                val_w_rhs = val_muc - val_Tmur
                val_w = _solve_delta_linear_system(
                    val_T, val_w_rhs, col, base_lane_id, active_mask
                )
                val_v = val_mur - _warp_reduce_sum_row(val_T * val_w, active_mask)
                val_D = (val_v + val_w - val_G) * val_T

                D[instance_id, row, col] = -val_D


def birkhoff_proj_n4_forward(
    R: torch.Tensor, tol: float = 1e-6
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("R", R)
    src_options = {"device": R.device, "dtype": R.dtype}
    R_work = _float32_contiguous(R)
    T_out = torch.empty_like(R_work)
    _birkhoff_proj_n4_forward_kernel(R_work, T_out, float(tol))
    return {"T": T_out.to(**src_options)}


def birkhoff_proj_n4_backward(
    G: torch.Tensor, T_proj: torch.Tensor
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("G", G)
    _check_n4_tensor("T_proj", T_proj)
    if G.shape != T_proj.shape:
        raise ValueError("G and T_proj must have the same shape")

    src_options = {"device": G.device, "dtype": G.dtype}
    G_work = _float32_contiguous(G)
    T_work = _float32_contiguous(T_proj)
    D = torch.empty_like(G_work)

    _birkhoff_proj_n4_backward_kernel(G_work, T_work, D)
    return {"D": D.to(**src_options)}


if __name__ == "__main__":
    R = torch.randn(128, 4, 4, device="cuda")
    # record time
    import time

    start_time = time.time()
    result = birkhoff_proj_n4_forward(R)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    print(result["T"].shape)
