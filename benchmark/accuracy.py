import torch
import triton
from mhc.kernels import _mhc_sinkhorn_fwd_kernel
import mhc_proj
from mhc.tilelang import (
    birkhoff_proj_n4_forward,
)
from mhc.tilelang.tilekernels import (
    sinkhorn_knopp_tilekernels_n4_forward,
)
from mhc.tilelang.tileexamples import (
    sinkhorn_knopp_tileexamples_n4_forward,
)


# Vanilla PyTorch implementation
@torch.compile
def vanilla(x, iters=20, eps=1e-6):
    P = torch.exp(x.float())
    for _ in range(iters):
        P = P / (P.sum(dim=2, keepdim=True) + eps)
        P = P / (P.sum(dim=1, keepdim=True) + eps)
    return P


# Triton-Sinkhorn: https://github.com/LottoLottoLotto/triton-sinkhorn
# Source included in the benchmark/mhc directory
# Derived from mhc.layer.MHCSinkhornFunction
@torch.compile
def fused(x, mhc_iters=20):
    W = x.contiguous()
    B, n, _ = W.shape
    M = torch.empty_like(W)

    NN = n * n
    BLOCK_SIZE = triton.next_power_of_2(NN)

    # History: [B, ITERS*2*NN] fp32 (row-phase + col-phase per iter)
    H = torch.empty((B, mhc_iters * 2 * NN), device=W.device, dtype=torch.float32)

    _mhc_sinkhorn_fwd_kernel[(B,)](
        W,
        M,
        H,
        W.stride(0),
        H.stride(0),
        N_LANES=n,
        ITERS=mhc_iters,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return M


# mHC.cu: https://github.com/AndreSlavescu/mHC.cu
# Source included in the src directory
def sk_n4(x, max_iter=20):
    expon = torch.exp(x)
    T = mhc_proj.torch.sinkhorn_knopp_n4(expon, max_iter)["T"]
    return T


# TileLang Sinkhorn-Knopp n=4 from deepseek TileKernels
def tl_tilekernels_n4(x, max_iter=20):
    T = sinkhorn_knopp_tilekernels_n4_forward(x.contiguous(), max_iter)["T"]
    return T


# TileLang Sinkhorn-Knopp n=4 from tile-lang examples
def tl_tileexamples_n4(x, max_iter=20):
    T = sinkhorn_knopp_tileexamples_n4_forward(x.contiguous(), max_iter)["T"]
    return T


# mHC-proj
def proj_n4(x, tol=1e-6):
    T = mhc_proj.torch.birkhoff_proj_n4(x, tol)["T"]
    return T


# TileLang mHC-proj
def tl_proj_n4(x, tol=1e-6):
    T = birkhoff_proj_n4_forward(x, tol)["T"]
    return T


# Marginal error of output 4x4 matrix
def marginal_error(T):
    row_sum = torch.sum(T.double(), dim=-1)
    col_sum = torch.sum(T.double(), dim=-2)
    err = (row_sum - 1.0).abs().sum(dim=-1) + (col_sum - 1.0).abs().sum(dim=-1)
    return err


# Print statistics
def print_error_stats(err, name):
    err_mean = err.mean().item()
    err_std = err.std().item()
    err_median = err.median().item()
    err_max = err.max().item()
    print(
        f"{name:15s}: mean={err_mean:.3e}, std={err_std:.3e}, median={err_median:.3e}, max={err_max:.3e}"
    )


# Accuracy test
def accuracy_test(x, input_distr):
    out_vanilla = vanilla(x)
    out_fused = fused(x)
    out_sk_n4 = sk_n4(x)
    out_tl_tilekernels_n4 = tl_tilekernels_n4(x)
    out_tl_tileexamples_n4 = tl_tileexamples_n4(x)
    out_proj_n4 = proj_n4(x)
    out_tl_proj_n4 = tl_proj_n4(x)

    print("=" * 80)
    print(f"Input Distribution: {input_distr}")
    print("=" * 80)
    print_error_stats(marginal_error(out_vanilla), "Vanilla")
    print_error_stats(marginal_error(out_fused), "Triton-Sinkhorn")
    print_error_stats(marginal_error(out_sk_n4), "mHC.cu")
    print_error_stats(marginal_error(out_tl_tilekernels_n4), "TL-TileKernels-n4")
    print_error_stats(marginal_error(out_tl_tileexamples_n4), "TL-TileExamples-n4")
    print_error_stats(marginal_error(out_proj_n4), "mHC-proj")
    print_error_stats(marginal_error(out_tl_proj_n4), "TL-mHC-proj")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N = 10000

    torch.manual_seed(123)
    x = torch.randn(N, 4, 4, device=device)
    accuracy_test(x, "N(0, 1)")

    torch.manual_seed(123)
    x = 10 * torch.randn(N, 4, 4, device=device)
    accuracy_test(x, "N(0, 10^2)")

    torch.manual_seed(123)
    x = 2 * torch.rand(N, 4, 4, device=device) - 1
    accuracy_test(x, "Unif(-1, 1)")

    torch.manual_seed(123)
    x = 20 * torch.rand(N, 4, 4, device=device) - 10
    accuracy_test(x, "Unif(-10, 10)")
