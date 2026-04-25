import tilelang
import tilelang.language as T
import torch


_N4 = 4
_EPS = 1e-8
_CG_ITERS = 2 * _N4
_THREADS_PER_BLOCK = 256
_THREADS_PER_INSTANCE = 16


def _check_n4_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3 or tensor.shape[-2:] != (_N4, _N4):
        raise ValueError(f"{name} must be a tensor of size B x 4 x 4")


def _cuda_float_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        tensor = tensor.to("cuda")
    return tensor.to(torch.float32).contiguous()


def _warp_reduce_sum_row(val, mask):
    val += T.shfl_xor(val, 1, mask=mask)
    val += T.shfl_xor(val, 2, mask=mask)
    return val


def _warp_reduce_max_row(val, mask):
    val = T.max(val, T.shfl_xor(val, 1, mask=mask))
    val = T.max(val, T.shfl_xor(val, 2, mask=mask))
    return val


def _warp_reduce_sum_col(val, mask):
    val += T.shfl_xor(val, 4, mask=mask)
    val += T.shfl_xor(val, 8, mask=mask)
    return val


def _warp_reduce_max_col(val, mask):
    val = T.max(val, T.shfl_xor(val, 4, mask=mask))
    val = T.max(val, T.shfl_xor(val, 8, mask=mask))
    return val


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def _sinkhorn_knopp_n4_forward_kernel(logits, out, max_iter: int):
    batch_size = T.dynamic("batch_size")

    logits: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    out: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore

    with T.Kernel(batch_size, threads=32) as b:
        cm = T.alloc_fragment((_N4, _N4), T.float32)
        row_sum = T.alloc_fragment(_N4, T.float32)
        col_sum = T.alloc_fragment(_N4, T.float32)
        row_max = T.alloc_fragment(_N4, T.float32)

        T.copy(logits[b, 0, 0], cm)

        T.reduce_max(cm, row_max, dim=1)
        for i, j in T.Parallel(_N4, _N4):
            cm[i, j] = T.exp(cm[i, j] - row_max[i])
        T.reduce_sum(cm, row_sum, dim=1)
        for i, j in T.Parallel(_N4, _N4):
            cm[i, j] = cm[i, j] / row_sum[i] + _EPS

        T.reduce_sum(cm, col_sum, dim=0)
        for i, j in T.Parallel(_N4, _N4):
            cm[i, j] = cm[i, j] / (col_sum[j] + _EPS)

        for _ in T.serial(max_iter - 1):
            T.reduce_sum(cm, row_sum, dim=1)
            for i, j in T.Parallel(_N4, _N4):
                cm[i, j] = cm[i, j] / (row_sum[i] + _EPS)

            T.reduce_sum(cm, col_sum, dim=0)
            for i, j in T.Parallel(_N4, _N4):
                cm[i, j] = cm[i, j] / (col_sum[j] + _EPS)

        T.copy(cm, out[b, 0, 0])


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def _sinkhorn_knopp_n4_backward_kernel(G, T_out, D, tilesize: int = 32):
    batch_size = T.dynamic("batch_size")

    G: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    T_out: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    D: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore

    with T.Kernel(T.ceildiv(batch_size, tilesize), threads=128) as tile_id:
        r = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        g = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        rg = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        d = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        x = T.alloc_fragment((tilesize, _N4), T.float32)
        y = T.alloc_fragment((tilesize, _N4), T.float32)
        res_x = T.alloc_fragment((tilesize, _N4), T.float32)
        res_y = T.alloc_fragment((tilesize, _N4), T.float32)
        p_x = T.alloc_fragment((tilesize, _N4), T.float32)
        p_y = T.alloc_fragment((tilesize, _N4), T.float32)
        ap_x = T.alloc_fragment((tilesize, _N4), T.float32)
        ap_y = T.alloc_fragment((tilesize, _N4), T.float32)
        tmp = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        dot_buf = T.alloc_fragment((tilesize, _N4), T.float32)
        r_norm = T.alloc_fragment(tilesize, T.float32)
        r_new_norm = T.alloc_fragment(tilesize, T.float32)
        p_ap = T.alloc_fragment(tilesize, T.float32)

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                r[t, i, j] = T_out[batch_idx, i, j]
                g[t, i, j] = G[batch_idx, i, j]
            else:
                r[t, i, j] = 0
                g[t, i, j] = 0

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            rg[t, i, j] = r[t, i, j] * g[t, i, j]
        T.reduce_sum(rg, res_x, dim=-1)
        T.reduce_sum(rg, res_y, dim=-2)

        T.clear(x)
        T.clear(y)
        T.copy(res_x, p_x)
        T.copy(res_y, p_y)

        for t, i in T.Parallel(tilesize, _N4):
            dot_buf[t, i] = res_x[t, i] * res_x[t, i] + res_y[t, i] * res_y[t, i]
        T.reduce_sum(dot_buf, r_norm, dim=-1)

        for _ in T.serial(_CG_ITERS):
            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                tmp[t, i, j] = r[t, i, j] * p_y[t, j]
            T.reduce_sum(tmp, ap_x, dim=-1)
            for t, i in T.Parallel(tilesize, _N4):
                ap_x[t, i] += p_x[t, i]

            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                tmp[t, i, j] = r[t, i, j] * p_x[t, i]
            T.reduce_sum(tmp, ap_y, dim=-2)
            for t, i in T.Parallel(tilesize, _N4):
                ap_y[t, i] += p_y[t, i]

            for t, i in T.Parallel(tilesize, _N4):
                dot_buf[t, i] = p_x[t, i] * ap_x[t, i] + p_y[t, i] * ap_y[t, i]
            T.reduce_sum(dot_buf, p_ap, dim=-1)

            for t, i in T.Parallel(tilesize, _N4):
                alpha = r_norm[t] / (p_ap[t] + _EPS)
                x[t, i] += alpha * p_x[t, i]
                y[t, i] += alpha * p_y[t, i]
                res_x[t, i] -= alpha * ap_x[t, i]
                res_y[t, i] -= alpha * ap_y[t, i]

            for t, i in T.Parallel(tilesize, _N4):
                dot_buf[t, i] = res_x[t, i] * res_x[t, i] + res_y[t, i] * res_y[t, i]
            T.reduce_sum(dot_buf, r_new_norm, dim=-1)

            for t, i in T.Parallel(tilesize, _N4):
                beta = r_new_norm[t] / (r_norm[t] + _EPS)
                p_x[t, i] = res_x[t, i] + beta * p_x[t, i]
                p_y[t, i] = res_y[t, i] + beta * p_y[t, i]

            T.copy(r_new_norm, r_norm)

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            d[t, i, j] = r[t, i, j] * (g[t, i, j] - x[t, i] - y[t, j])

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                D[batch_idx, i, j] = d[t, i, j]


@tilelang.jit
def _birkhoff_proj_n4_forward_kernel(R, T_out, tol: float = 1e-6):
    N = T.dynamic("N")
    dtype = T.float32

    R: T.Tensor((N, _N4, _N4), dtype)  # type: ignore
    T_out: T.Tensor((N, _N4, _N4), dtype)  # type: ignore

    numBlocks = T.ceildiv(N * _THREADS_PER_INSTANCE, _THREADS_PER_BLOCK)

    with T.Kernel(numBlocks, threads=_THREADS_PER_BLOCK) as (bx,):
        for tx in T.Parallel(_THREADS_PER_BLOCK):
            global_tid = bx * _THREADS_PER_BLOCK + tx
            instance_id = global_tid // _THREADS_PER_INSTANCE

            if instance_id < N:
                lane_id = tx & 31
                lane_id_gr = tx % _THREADS_PER_INSTANCE
                base_lane_id = lane_id & (~15)
                active_mask = T.if_then_else(lane_id < 16, 0x0000FFFF, 0xFFFF0000)

                row = lane_id_gr // 4
                col = lane_id_gr % 4
                ind_mat = instance_id * (_N4 * _N4) + lane_id_gr

                val_R = R[instance_id, row, col]
                row_sum_R = _warp_reduce_sum_row(val_R, active_mask)
                row_max_R = _warp_reduce_max_row(val_R, active_mask)
                col_sum_R = _warp_reduce_sum_col(val_R, active_mask)
                col_max_R = _warp_reduce_max_col(val_R, active_mask)


def birkhoff_proj_n4_forward(
    R: torch.Tensor, tol: float = 1e-6
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("R", R)
    src_options = {"device": R.device, "dtype": R.dtype}
    R_work = _cuda_float_contiguous(R)
    T_out = torch.empty_like(R_work)
    _birkhoff_proj_n4_forward_kernel(R_work, T_out, tol)
    return {"T": T_out.to(**src_options)}


def birkhoff_proj_n4_backward(
    G: torch.Tensor, T: torch.Tensor
) -> dict[str, torch.Tensor]: ...


def sinkhorn_knopp_n4_forward(
    R: torch.Tensor, max_iter: int = 20
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("R", R)
    src_options = {"device": R.device, "dtype": R.dtype}
    R_work = _cuda_float_contiguous(R)
    T_out = torch.empty_like(R_work)

    _sinkhorn_knopp_n4_forward_kernel(R_work, T_out, max_iter)

    return {"T": T_out.to(**src_options)}


def sinkhorn_knopp_n4_backward(
    G: torch.Tensor, logits: torch.Tensor, max_iter: int = 20
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("G", G)
    _check_n4_tensor("logits", logits)
    if G.shape != logits.shape:
        raise ValueError("G and logits must have the same shape")

    src_options = {"device": G.device, "dtype": G.dtype}
    G_work = _cuda_float_contiguous(G)
    logits_work = _cuda_float_contiguous(logits)
    T_out = torch.empty_like(logits_work)
    D = torch.empty_like(G_work)

    _sinkhorn_knopp_n4_forward_kernel(logits_work, T_out, max_iter)
    _sinkhorn_knopp_n4_backward_kernel(G_work, T_out, D)

    return {"D": D.to(**src_options)}


if __name__ == "__main__":
    # Example usage
    R = torch.randn(10, 4, 4).cuda()
    result = birkhoff_proj_n4_forward(R)
