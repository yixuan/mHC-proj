# Adapted from tile-lang example:
# https://github.com/tile-ai/tilelang/tree/8f4a08f56de7683162f5a84fdae7be3a5d98d8e2/examples/deepseek_mhc

import tilelang
import tilelang.language as T
import torch


_N4 = 4
_CG_EPS = 1e-10
_CG_ITERS = 2 * _N4


def _check_n4_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3 or tensor.shape[-2:] != (_N4, _N4):
        raise ValueError(f"{name} must be a tensor of size B x 4 x 4")


def _cuda_float_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        tensor = tensor.to("cuda")
    return tensor.to(torch.float32).contiguous()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def _sinkhorn_knopp_tileexamples_n4_forward_kernel(
    R, out, max_iter: int, eps: float
):
    batch_size = T.dynamic("batch_size")

    R: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    out: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore

    with T.Kernel(batch_size, threads=32) as b:
        cm = T.alloc_fragment((_N4, _N4), T.float32)
        row_sum = T.alloc_fragment(_N4, T.float32)
        col_sum = T.alloc_fragment(_N4, T.float32)
        row_max = T.alloc_fragment(_N4, T.float32)

        T.copy(R[b, 0, 0], cm)

        T.reduce_max(cm, row_max, dim=1)
        for i, j in T.Parallel(_N4, _N4):
            cm[i, j] = T.exp(cm[i, j] - row_max[i])
        T.reduce_sum(cm, row_sum, dim=1)
        for i, j in T.Parallel(_N4, _N4):
            cm[i, j] = cm[i, j] / row_sum[i] + eps

        T.reduce_sum(cm, col_sum, dim=0)
        for i, j in T.Parallel(_N4, _N4):
            cm[i, j] = cm[i, j] / (col_sum[j] + eps)

        for _ in T.serial(max_iter - 1):
            T.reduce_sum(cm, row_sum, dim=1)
            for i, j in T.Parallel(_N4, _N4):
                cm[i, j] = cm[i, j] / (row_sum[i] + eps)

            T.reduce_sum(cm, col_sum, dim=0)
            for i, j in T.Parallel(_N4, _N4):
                cm[i, j] = cm[i, j] / (col_sum[j] + eps)

        T.copy(cm, out[b, 0, 0])


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def _sinkhorn_knopp_tileexamples_n4_backward_kernel(
    G, T_out, D, tilesize: int = 16
):
    batch_size = T.dynamic("batch_size")

    G: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    T_out: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    D: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore

    with T.Kernel(T.ceildiv(batch_size, tilesize), threads=128) as tile_id:
        r = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        g = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        rg = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        d = T.alloc_shared((tilesize, _N4, _N4), T.float32)
        x1 = T.alloc_shared((tilesize, _N4), T.float32)
        x2 = T.alloc_shared((tilesize, _N4), T.float32)
        r1 = T.alloc_shared((tilesize, _N4), T.float32)
        r2 = T.alloc_shared((tilesize, _N4), T.float32)
        p1 = T.alloc_shared((tilesize, _N4), T.float32)
        p2 = T.alloc_shared((tilesize, _N4), T.float32)
        ap1 = T.alloc_shared((tilesize, _N4), T.float32)
        ap2 = T.alloc_shared((tilesize, _N4), T.float32)
        alpha = T.alloc_fragment((tilesize, _N4), T.float32)
        beta = T.alloc_fragment((tilesize, _N4), T.float32)
        r_normsq = T.alloc_fragment(tilesize, T.float32)
        r_new_normsq = T.alloc_fragment(tilesize, T.float32)
        p_ap = T.alloc_fragment(tilesize, T.float32)
        buf1 = T.alloc_shared((tilesize, _N4, _N4), T.float32)
        buf2 = T.alloc_shared((tilesize, _N4), T.float32)

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                r[t, i, j] = T_out[batch_idx, i, j]
                g[t, i, j] = G[batch_idx, i, j]
            else:
                r[t, i, j] = 0.0
                g[t, i, j] = 0.0

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            rg[t, i, j] = r[t, i, j] * g[t, i, j]
        T.reduce_sum(rg, r1, dim=-1)
        T.reduce_sum(rg, r2, dim=-2)

        T.fill(x1, 0.0)
        T.fill(x2, 0.0)
        T.copy(r1, p1)
        T.copy(r2, p2)

        for t, i in T.Parallel(tilesize, _N4):
            buf2[t, i] = r1[t, i] * r1[t, i] + r2[t, i] * r2[t, i]
        T.reduce_sum(buf2, r_normsq, dim=-1)

        for _ in T.serial(_CG_ITERS):
            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                buf1[t, i, j] = r[t, i, j] * p2[t, j]
            T.reduce_sum(buf1, ap1, dim=-1)
            for t, i in T.Parallel(tilesize, _N4):
                ap1[t, i] += p1[t, i]

            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                buf1[t, i, j] = r[t, i, j] * p1[t, i]
            T.reduce_sum(buf1, ap2, dim=-2)
            for t, i in T.Parallel(tilesize, _N4):
                ap2[t, i] += p2[t, i]

            for t, i in T.Parallel(tilesize, _N4):
                buf2[t, i] = p1[t, i] * ap1[t, i] + p2[t, i] * ap2[t, i]
            T.reduce_sum(buf2, p_ap, dim=-1)

            for t, i in T.Parallel(tilesize, _N4):
                alpha[t, i] = r_normsq[t] / (p_ap[t] + _CG_EPS)
                x1[t, i] += alpha[t, i] * p1[t, i]
                x2[t, i] += alpha[t, i] * p2[t, i]
                r1[t, i] -= alpha[t, i] * ap1[t, i]
                r2[t, i] -= alpha[t, i] * ap2[t, i]

            for t, i in T.Parallel(tilesize, _N4):
                buf2[t, i] = r1[t, i] * r1[t, i] + r2[t, i] * r2[t, i]
            T.reduce_sum(buf2, r_new_normsq, dim=-1)

            for t, i in T.Parallel(tilesize, _N4):
                beta[t, i] = r_new_normsq[t] / (r_normsq[t] + _CG_EPS)
                p1[t, i] = r1[t, i] + beta[t, i] * p1[t, i]
                p2[t, i] = r2[t, i] + beta[t, i] * p2[t, i]

            T.copy(r_new_normsq, r_normsq)

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            d[t, i, j] = r[t, i, j] * (g[t, i, j] - x1[t, i] - x2[t, j])

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                D[batch_idx, i, j] = d[t, i, j]


def sinkhorn_knopp_tileexamples_n4_forward(
    R: torch.Tensor, max_iter: int = 20, eps: float = 1e-6
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("R", R)
    src_options = {"device": R.device, "dtype": R.dtype}
    R_work = _cuda_float_contiguous(R)
    T_out = torch.empty_like(R_work)

    _sinkhorn_knopp_tileexamples_n4_forward_kernel(R_work, T_out, max_iter, eps)

    return {"T": T_out.to(**src_options)}


def sinkhorn_knopp_tileexamples_n4_backward(
    G: torch.Tensor, T_out: torch.Tensor
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("G", G)
    _check_n4_tensor("T_out", T_out)
    if G.shape != T_out.shape:
        raise ValueError("G and T_out must have the same shape")

    src_options = {"device": G.device, "dtype": G.dtype}
    G_work = _cuda_float_contiguous(G)
    T_work = _cuda_float_contiguous(T_out)
    D = torch.empty_like(G_work)

    _sinkhorn_knopp_tileexamples_n4_backward_kernel(G_work, T_work, D)

    return {"D": D.to(**src_options)}
