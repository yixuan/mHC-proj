import tilelang
import tilelang.language as T
import torch


_N4 = 4
EPS = 1e-10


def _check_square_tensor(name: str, tensor: torch.Tensor, n: int) -> None:
    if tensor.ndim != 3 or tensor.shape[-2:] != (n, n):
        raise ValueError(f"{name} must be a tensor of size B x {n} x {n}")


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
def _mhc_sinkhorn_fwd_tilelang(
    comb_mix,
    comb_mix_out,
    sinkhorn_repeat: int,
    hc_sinkhorn_eps: float,
    hc_mult: int = 4,
):
    """Standalone Sinkhorn kernel adapted from example_mhc_pre.py."""
    num_tokens = T.dynamic("num_tokens")

    comb_mix: T.Tensor((num_tokens, hc_mult, hc_mult), T.float32)  # type: ignore
    comb_mix_out: T.Tensor((num_tokens, hc_mult, hc_mult), T.float32)  # type: ignore

    with T.Kernel(num_tokens, threads=32) as i:
        cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
        row_sum = T.alloc_fragment(hc_mult, T.float32)
        col_sum = T.alloc_fragment(hc_mult, T.float32)

        T.copy(comb_mix[i, 0, 0], cm)

        # comb = comb.softmax(-1) + eps
        row_max = T.alloc_fragment(hc_mult, T.float32)
        T.reduce_max(cm, row_max, dim=1)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = T.exp(cm[j, k] - row_max[j])
        T.reduce_sum(cm, row_sum, dim=1)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

        # comb = comb / (comb.sum(-2) + eps)
        T.reduce_sum(cm, col_sum, dim=0)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

        for _ in T.serial(sinkhorn_repeat - 1):
            # comb = comb / (comb.sum(-1) + eps)
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

        T.copy(cm, comb_mix_out[i, 0, 0])


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def _mhc_sinkhorn_bwd_implicit_cg_tilelang(
    n_stream: int = 4,
    tilesize: int = 16,
    threads: int = 128,
):
    """Standalone implicit-CG backward adapted from example_mhc_bwd.py."""
    seqlen = T.dynamic("seqlen")
    tensor_shape = [seqlen, n_stream, n_stream]
    dtype = T.float32

    @T.macro
    def matvec_A(R, x1, x2, buf, y1, y2):
        for i_tile, i, j in T.Parallel(tilesize, n_stream, n_stream):
            buf[i_tile, i, j] = R[i_tile, i, j] * x2[i_tile, j]
        T.reduce_sum(buf, y1, dim=-1)

        for i_tile, i, j in T.Parallel(tilesize, n_stream, n_stream):
            buf[i_tile, i, j] = R[i_tile, i, j] * x1[i_tile, i]
        T.reduce_sum(buf, y2, dim=-2)

        for i_tile, i in T.Parallel(tilesize, n_stream):
            y1[i_tile, i] += x1[i_tile, i]
            y2[i_tile, i] += x2[i_tile, i]

    @T.macro
    def dot(x1, x2, y1, y2, buf, out):
        for i_tile, i in T.Parallel(tilesize, n_stream):
            buf[i_tile, i] = (
                x1[i_tile, i] * y1[i_tile, i]
                + x2[i_tile, i] * y2[i_tile, i]
            )

        T.reduce_sum(buf, out, dim=-1)

    @T.prim_func
    def main(
        out: T.Tensor(tensor_shape, dtype),
        dout: T.Tensor(tensor_shape, dtype),
        res: T.Tensor(tensor_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seqlen, tilesize), threads=threads) as i_seq:
            R = T.alloc_fragment([tilesize, n_stream, n_stream], dtype=dtype)
            dR = T.alloc_fragment([tilesize, n_stream, n_stream], dtype=dtype)
            RdR = T.alloc_fragment([tilesize, n_stream, n_stream], dtype=dtype)
            res_tile = T.alloc_shared([tilesize, n_stream, n_stream], dtype=dtype)
            b1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            b2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            x1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            x2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            r1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            r2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            p1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            p2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            alpha = T.alloc_fragment([tilesize, n_stream], dtype=dtype)
            beta = T.alloc_fragment([tilesize, n_stream], dtype=dtype)
            r_normsq = T.alloc_fragment([tilesize], dtype=dtype)
            r_new_normsq = T.alloc_fragment([tilesize], dtype=dtype)
            Ap1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            Ap2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            pAp = T.alloc_fragment([tilesize], dtype=dtype)

            # Buffers for intermediate results
            buf1 = T.alloc_shared([tilesize, n_stream, n_stream], dtype=dtype)
            buf2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)

            for i_tile, i_nx, i_ny in T.Parallel(tilesize, n_stream, n_stream):
                token_idx = i_seq * tilesize + i_tile
                if token_idx < seqlen:
                    R[i_tile, i_nx, i_ny] = out[token_idx, i_nx, i_ny]
                    dR[i_tile, i_nx, i_ny] = dout[token_idx, i_nx, i_ny]
                else:
                    R[i_tile, i_nx, i_ny] = 0.0
                    dR[i_tile, i_nx, i_ny] = 0.0

            for i_tile, i_nx, i_ny in T.Parallel(tilesize, n_stream, n_stream):
                RdR[i_tile, i_nx, i_ny] = R[i_tile, i_nx, i_ny] * dR[i_tile, i_nx, i_ny]

            T.reduce_sum(RdR, b1, dim=-1)
            T.reduce_sum(RdR, b2, dim=-2)

            T.fill(x1, 0.0)
            T.fill(x2, 0.0)

            matvec_A(R, x1, x2, buf1, r1, r2)

            for i_tile, i_n in T.Parallel(tilesize, n_stream):
                r1[i_tile, i_n] = b1[i_tile, i_n] - r1[i_tile, i_n]

            for i_tile, i_n in T.Parallel(tilesize, n_stream):
                r2[i_tile, i_n] = b2[i_tile, i_n] - r2[i_tile, i_n]

            T.copy(r1, p1)
            T.copy(r2, p2)

            dot(r1, r2, r1, r2, buf2, r_normsq)

            # Conjugate gradient: iteration starts
            for _ in T.serial(2 * n_stream):
                matvec_A(R, p1, p2, buf1, Ap1, Ap2)

                dot(p1, p2, Ap1, Ap2, buf2, pAp)

                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    alpha[i_tile, i_n] = r_normsq[i_tile] / (pAp[i_tile] + EPS)
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    x1[i_tile, i_n] += alpha[i_tile, i_n] * p1[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    x2[i_tile, i_n] += alpha[i_tile, i_n] * p2[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    r1[i_tile, i_n] -= alpha[i_tile, i_n] * Ap1[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    r2[i_tile, i_n] -= alpha[i_tile, i_n] * Ap2[i_tile, i_n]

                dot(r1, r2, r1, r2, buf2, r_new_normsq)

                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    beta[i_tile, i_n] = r_new_normsq[i_tile] / (
                        r_normsq[i_tile] + EPS
                    )
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    p1[i_tile, i_n] = r1[i_tile, i_n] + beta[i_tile, i_n] * p1[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    p2[i_tile, i_n] = r2[i_tile, i_n] + beta[i_tile, i_n] * p2[i_tile, i_n]

                T.copy(r_new_normsq, r_normsq)
            # Conjugate gradient: iteration ends

            for i_tile, i_nx, i_ny in T.Parallel(tilesize, n_stream, n_stream):
                res_tile[i_tile, i_nx, i_ny] = (
                    dR[i_tile, i_nx, i_ny] - x1[i_tile, i_nx] - x2[i_tile, i_ny]
                ) * R[i_tile, i_nx, i_ny]

            for i_tile, i_nx, i_ny in T.Parallel(tilesize, n_stream, n_stream):
                token_idx = i_seq * tilesize + i_tile
                if token_idx < seqlen:
                    res[token_idx, i_nx, i_ny] = res_tile[i_tile, i_nx, i_ny]

    return main


def sinkhorn_knopp_tileexamples_n4_forward(
    R: torch.Tensor, max_iter: int = 20, eps: float = 1e-6
) -> dict[str, torch.Tensor]:
    _check_square_tensor("R", R, _N4)
    src_options = {"device": R.device, "dtype": R.dtype}
    comb_mix = _cuda_float_contiguous(R)
    comb_mix_out = torch.empty_like(comb_mix)

    _mhc_sinkhorn_fwd_tilelang(comb_mix, comb_mix_out, max_iter, eps, _N4)

    return {"T": comb_mix_out.to(**src_options)}


def sinkhorn_knopp_tileexamples_n4_backward(
    G: torch.Tensor, T_out: torch.Tensor
) -> dict[str, torch.Tensor]:
    _check_square_tensor("G", G, _N4)
    _check_square_tensor("T_out", T_out, _N4)
    if G.shape != T_out.shape:
        raise ValueError("G and T_out must have the same shape")

    src_options = {"device": G.device, "dtype": G.dtype}
    dout = _cuda_float_contiguous(G)
    out = _cuda_float_contiguous(T_out)
    res = torch.empty_like(dout)

    kernel = _mhc_sinkhorn_bwd_implicit_cg_tilelang(_N4)
    kernel(out, dout, res)

    return {"D": res.to(**src_options)}
