# Adapted from official sinkhorn kernel:
# https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/sinkhorn_kernel.py

import tilelang
import tilelang.language as T
import torch


_N4 = 4
_EPS = 1e-8


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
    },
)
def _sinkhorn_knopp1_n4_forward_kernel(
    R, out, max_iter: int, eps: float, tilesize: int = 16
):
    batch_size = T.dynamic("batch_size")

    R: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    out: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore

    with T.Kernel(T.ceildiv(batch_size, tilesize)) as tile_id:
        cm = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        row_sum = T.alloc_fragment((tilesize, _N4), T.float32)
        col_sum = T.alloc_fragment((tilesize, _N4), T.float32)
        row_max = T.alloc_fragment((tilesize, _N4), T.float32)

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                cm[t, i, j] = R[batch_idx, i, j]
            else:
                cm[t, i, j] = 0.0

        T.reduce_max(cm, row_max, dim=2)
        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            cm[t, i, j] = T.exp(cm[t, i, j] - row_max[t, i])
        T.reduce_sum(cm, row_sum, dim=2)
        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            cm[t, i, j] = cm[t, i, j] / row_sum[t, i] + eps

        T.reduce_sum(cm, col_sum, dim=1)
        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            cm[t, i, j] = cm[t, i, j] / (col_sum[t, j] + eps)

        for _ in T.serial(max_iter - 1):
            T.reduce_sum(cm, row_sum, dim=2)
            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                cm[t, i, j] = cm[t, i, j] / (row_sum[t, i] + eps)

            T.reduce_sum(cm, col_sum, dim=1)
            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                cm[t, i, j] = cm[t, i, j] / (col_sum[t, j] + eps)

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                out[batch_idx, i, j] = cm[t, i, j]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _sinkhorn_knopp1_n4_backward_kernel(
    G, R, D, max_iter: int, eps: float, tilesize: int = 16
):
    batch_size = T.dynamic("batch_size")

    G: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    R: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore
    D: T.Tensor((batch_size, _N4, _N4), T.float32)  # type: ignore

    with T.Kernel(T.ceildiv(batch_size, tilesize), threads=128) as tile_id:
        grad = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        x = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        x_inter = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        temp = T.alloc_fragment((tilesize, _N4, _N4), T.float32)
        row_sum = T.alloc_fragment((tilesize, _N4), T.float32)
        row_sum2 = T.alloc_fragment((tilesize, _N4), T.float32)
        col_sum = T.alloc_fragment((tilesize, _N4), T.float32)
        col_sum2 = T.alloc_fragment((tilesize, _N4), T.float32)
        row_max = T.alloc_fragment((tilesize, _N4), T.float32)
        xs = T.alloc_shared((max_iter * 2, tilesize, _N4, _N4), T.float32)
        sums = T.alloc_shared((max_iter * 2, tilesize, _N4), T.float32)

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                x[t, i, j] = R[batch_idx, i, j]
                grad[t, i, j] = G[batch_idx, i, j]
            else:
                x[t, i, j] = 0.0
                grad[t, i, j] = 0.0

        T.reduce_max(x, row_max, dim=2)
        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            x[t, i, j] = T.exp(x[t, i, j] - row_max[t, i])
        T.reduce_sum(x, row_sum, dim=2)
        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            x[t, i, j] = x[t, i, j] / row_sum[t, i]
            xs[0, t, i, j] = x[t, i, j]
            x[t, i, j] = x[t, i, j] + eps
            xs[1, t, i, j] = x[t, i, j]

        T.reduce_sum(x, col_sum, dim=1)
        for t, j in T.Parallel(tilesize, _N4):
            sums[1, t, j] = col_sum[t, j]
        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            x[t, i, j] = x[t, i, j] / (col_sum[t, j] + eps)

        for step in T.serial(max_iter - 1):
            T.reduce_sum(x, row_sum, dim=2)
            for t, i in T.Parallel(tilesize, _N4):
                sums[step * 2 + 2, t, i] = row_sum[t, i]
            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                xs[step * 2 + 2, t, i, j] = x[t, i, j]
                x[t, i, j] = x[t, i, j] / (row_sum[t, i] + eps)

            T.reduce_sum(x, col_sum, dim=1)
            for t, j in T.Parallel(tilesize, _N4):
                sums[step * 2 + 3, t, j] = col_sum[t, j]
            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                xs[step * 2 + 3, t, i, j] = x[t, i, j]
                x[t, i, j] = x[t, i, j] / (col_sum[t, j] + eps)

        for inv_step in T.serial(2 * max_iter - 1):
            step_idx = 2 * max_iter - 1 - inv_step
            for t, i, j in T.Parallel(tilesize, _N4, _N4):
                x_inter[t, i, j] = xs[step_idx, t, i, j]
                temp[t, i, j] = grad[t, i, j] * x_inter[t, i, j]

            if inv_step % 2 == 0:
                T.reduce_sum(temp, col_sum2, dim=1)
                for t, j in T.Parallel(tilesize, _N4):
                    col_sum[t, j] = sums[step_idx, t, j]
                    col_sum2[t, j] = col_sum2[t, j] / (col_sum[t, j] + eps)
                for t, i, j in T.Parallel(tilesize, _N4, _N4):
                    grad[t, i, j] = (grad[t, i, j] - col_sum2[t, j]) / (
                        col_sum[t, j] + eps
                    )
            else:
                T.reduce_sum(temp, row_sum2, dim=2)
                for t, i in T.Parallel(tilesize, _N4):
                    row_sum[t, i] = sums[step_idx, t, i]
                    row_sum2[t, i] = row_sum2[t, i] / (row_sum[t, i] + eps)
                for t, i, j in T.Parallel(tilesize, _N4, _N4):
                    grad[t, i, j] = (grad[t, i, j] - row_sum2[t, i]) / (
                        row_sum[t, i] + eps
                    )

        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            x_inter[t, i, j] = xs[0, t, i, j]
            temp[t, i, j] = grad[t, i, j] * x_inter[t, i, j]
        T.reduce_sum(temp, row_sum, dim=2)
        for t, i, j in T.Parallel(tilesize, _N4, _N4):
            batch_idx = tile_id * tilesize + t
            if batch_idx < batch_size:
                D[batch_idx, i, j] = (grad[t, i, j] - row_sum[t, i]) * x_inter[t, i, j]


def sinkhorn_knopp1_n4_forward(
    R: torch.Tensor, max_iter: int = 20, eps: float = 1e-6
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("R", R)
    src_options = {"device": R.device, "dtype": R.dtype}
    R_work = _cuda_float_contiguous(R)
    T_out = torch.empty_like(R_work)

    _sinkhorn_knopp1_n4_forward_kernel(R_work, T_out, max_iter, eps)

    return {"T": T_out.to(**src_options)}


def sinkhorn_knopp1_n4_backward(
    G: torch.Tensor, R: torch.Tensor, max_iter: int = 20, eps: float = 1e-6
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("G", G)
    _check_n4_tensor("R", R)
    if G.shape != R.shape:
        raise ValueError("G and R must have the same shape")

    src_options = {"device": G.device, "dtype": G.dtype}
    G_work = _cuda_float_contiguous(G)
    R_work = _cuda_float_contiguous(R)
    D = torch.empty_like(G_work)

    _sinkhorn_knopp1_n4_backward_kernel(G_work, R_work, D, max_iter, eps)

    return {"D": D.to(**src_options)}
