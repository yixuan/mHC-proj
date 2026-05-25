import pandas as pd
import torch
import triton
from mhc.kernels import _mhc_sinkhorn_fwd_kernel, _mhc_sinkhorn_bwd_kernel
import mhc_proj
from mhc.tilelang import (
    birkhoff_proj_n4_forward,
    birkhoff_proj_n4_backward,
)
from mhc.tilelang.tilekernels import (
    sinkhorn_knopp_tilekernels_n4_forward,
    sinkhorn_knopp_tilekernels_n4_backward,
)
from mhc.tilelang.tileexamples import (
    sinkhorn_knopp_tileexamples_n4_forward,
    sinkhorn_knopp_tileexamples_n4_backward,
)


# Vanilla PyTorch implementation
@torch.compile
def vanilla_fwd(x, iters=20, eps=1e-6):
    P = torch.exp(x)
    for _ in range(iters):
        P = P / (P.sum(dim=2, keepdim=True) + eps)
        P = P / (P.sum(dim=1, keepdim=True) + eps)
    return P, None


@torch.compile
def vanilla_fwd_bwd(x, G, iters=20, eps=1e-6):
    out, *saved = vanilla_fwd(x, iters, eps)
    torch.autograd.backward(out, grad_tensors=G)
    return out.detach(), x.grad


# Triton-Sinkhorn: https://github.com/LottoLottoLotto/triton-sinkhorn
# Source included in the benchmark/mhc directory
# Derived from mhc.layer.MHCSinkhornFunction
@torch.compile
def fused_fwd(x, mhc_iters=20):
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
    return M, W, H


@torch.compile
def fused_bwd(G, W, H, mhc_iters=20):
    grad_output = G.contiguous()

    grad_W = torch.empty_like(W)
    B, n, iters = W.shape[0], 4, mhc_iters

    NN = n * n
    BLOCK_SIZE = triton.next_power_of_2(NN)

    _mhc_sinkhorn_bwd_kernel[(B,)](
        grad_output,
        W,
        H,
        grad_W,
        W.stride(0),
        H.stride(0),
        N_LANES=n,
        ITERS=iters,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_W


def fused_fwd_bwd(x, G, mhc_iters=20):
    out, *saved = fused_fwd(x, mhc_iters)
    grad = fused_bwd(G, *saved, mhc_iters=mhc_iters)
    return out.detach(), grad


# mHC.cu: https://github.com/AndreSlavescu/mHC.cu
# Source included in the src directory
def sk_n4_fwd(x, max_iter=20):
    expon = torch.exp(x)
    return mhc_proj.torch.sinkhorn_knopp_n4(expon, max_iter)["T"], expon


def sk_n4_bwd(G, expM, max_iter=20):
    return mhc_proj.torch.sinkhorn_knopp_n4_backward(G, expM, max_iter)["D"] * expM


def sk_n4_fwd_bwd(x, G, max_iter=20):
    out, *saved = sk_n4_fwd(x, max_iter)
    grad = sk_n4_bwd(G, *saved, max_iter=max_iter)
    return out.detach(), grad


# TileLang Sinkhorn-Knopp n=4 from deepseek TileKernels
def tl_tilekernels_n4_fwd(x, max_iter=20):
    R = x.contiguous()
    return sinkhorn_knopp_tilekernels_n4_forward(R, max_iter)["T"], R


def tl_tilekernels_n4_bwd(G, R, max_iter=20):
    return sinkhorn_knopp_tilekernels_n4_backward(G, R, max_iter)["D"]


def tl_tilekernels_n4_fwd_bwd(x, G, max_iter=20):
    out, *saved = tl_tilekernels_n4_fwd(x, max_iter)
    grad = tl_tilekernels_n4_bwd(G, *saved, max_iter=max_iter)
    return out.detach(), grad


# TileLang Sinkhorn-Knopp n=4 from tile-lang examples
def tl_tileexamples_n4_fwd(x, max_iter=20):
    R = x.contiguous()
    T = sinkhorn_knopp_tileexamples_n4_forward(R, max_iter)["T"]
    return T, T


def tl_tileexamples_n4_bwd(G, T):
    return sinkhorn_knopp_tileexamples_n4_backward(G, T)["D"]


def tl_tileexamples_n4_fwd_bwd(x, G, max_iter=20):
    out, *saved = tl_tileexamples_n4_fwd(x, max_iter)
    grad = tl_tileexamples_n4_bwd(G, *saved)
    return out.detach(), grad


# mHC-proj
def proj_n4_fwd(x, tol=1e-6):
    T = mhc_proj.torch.birkhoff_proj_n4(x, tol)["T"]
    return T, T


def proj_n4_bwd(G, T):
    return mhc_proj.torch.birkhoff_proj_n4_backward(G, T)["D"]


def proj_n4_fwd_bwd(x, G, tol=1e-6):
    out, *saved = proj_n4_fwd(x, tol)
    grad = proj_n4_bwd(G, *saved)
    return out.detach(), grad


# TileLang mHC-proj
def tl_proj_n4_fwd(x, tol=1e-6):
    T = birkhoff_proj_n4_forward(x, tol)["T"]
    return T, T


def tl_proj_n4_bwd(G, T):
    return birkhoff_proj_n4_backward(G, T)["D"]


def tl_proj_n4_fwd_bwd(x, G, tol=1e-6):
    out, *saved = tl_proj_n4_fwd(x, tol)
    grad = tl_proj_n4_bwd(G, *saved)
    return out.detach(), grad


def benchmark(fn, x, G, iters=100, backward=True):
    # Warmup
    xc = x.clone().detach()
    if backward:
        for _ in range(100):
            xc.requires_grad_(True)
            xc, grad = fn(xc, G)
    else:
        with torch.no_grad():
            for _ in range(100):
                xc, *saved = fn(xc)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Benchmark
    xc = x.clone().detach()
    start.record()
    if backward:
        for _ in range(iters):
            xc.requires_grad_(True)
            xc, grad = fn(xc, G)
    else:
        with torch.no_grad():
            for _ in range(iters):
                xc, *saved = fn(xc)
    end.record()
    torch.cuda.synchronize()
    t_ms = start.elapsed_time(end)
    return t_ms


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        {"N": 128, "n": 4},
        {"N": 512, "n": 4},
        {"N": 2048, "n": 4},
        {"N": 8192, "n": 4},
        {"N": 32768, "n": 4},
        {"N": 131072, "n": 4},
    ]
    nruns = 10

    torch.manual_seed(123)
    results = []
    print(f"Benchmarking on {torch.cuda.get_device_name(0)}...")
    for cfg in configs:
        N, n = cfg["N"], cfg["n"]
        print(f"Testing on N = {N}, n = {n}...")
        x = 10 * torch.randn(N, n, n, device=device)
        G = torch.randn(N, n, n, device=device)

        for run in range(nruns):
            # print(f"N = {N}, n = {n}, run {run}")
            x = x.clone().detach()
            G = G.clone().detach()

            t_vanilla = benchmark(vanilla_fwd, x, G, iters=100, backward=False)
            t_fused = benchmark(fused_fwd, x, G, iters=100, backward=False)
            t_sk_n4 = benchmark(sk_n4_fwd, x, G, iters=100, backward=False)
            t_tl_tilekernels_n4 = benchmark(
                tl_tilekernels_n4_fwd, x, G, iters=100, backward=False
            )
            t_tl_tileexamples_n4 = benchmark(
                tl_tileexamples_n4_fwd, x, G, iters=100, backward=False
            )
            t_proj_n4 = benchmark(proj_n4_fwd, x, G, iters=100, backward=False)
            t_tl_proj_n4 = benchmark(tl_proj_n4_fwd, x, G, iters=100, backward=False)

            results.append({
                "N": N,
                "n": n,
                "Vanilla": t_vanilla / t_proj_n4,
                "Fused": t_fused / t_proj_n4,
                "SK-n4": t_sk_n4 / t_proj_n4,
                "TL-TileKernels-n4": t_tl_tilekernels_n4 / t_proj_n4,
                "TL-TileExamples-n4": t_tl_tileexamples_n4 / t_proj_n4,
                "TL-Proj-n4": t_tl_proj_n4 / t_proj_n4,
                "Proj-n4": t_proj_n4 / t_proj_n4,
            })
    dat = pd.DataFrame(results)
    print("=" * 70)
    print("Forward Pass Only")
    print("=" * 70)
    print("Mean Run Time")
    print("=" * 70)
    print(dat.groupby(["N", "n"], as_index=False).mean().to_string(index=False))
    print("=" * 70)
    print("Median Run Time")
    print("=" * 70)
    print(dat.groupby(["N", "n"], as_index=False).median().to_string(index=False))
    print()

    torch.manual_seed(123)
    results = []
    for cfg in configs:
        N, n = cfg["N"], cfg["n"]
        print(f"Testing on N = {N}, n = {n}...")
        x = torch.randn(N, n, n, device=device)
        G = torch.randn(N, n, n, device=device)

        for run in range(nruns):
            # print(f"N = {N}, n = {n}, run {run}")
            x = x.clone().detach().requires_grad_(True)
            G = G.clone().detach()

            t_vanilla = benchmark(vanilla_fwd_bwd, x, G, iters=100, backward=True)
            t_fused = benchmark(fused_fwd_bwd, x, G, iters=100, backward=True)
            t_sk_n4 = benchmark(sk_n4_fwd_bwd, x, G, iters=100, backward=True)
            t_tl_tilekernels_n4 = benchmark(
                tl_tilekernels_n4_fwd_bwd, x, G, iters=100, backward=True
            )
            t_tl_tileexamples_n4 = benchmark(
                tl_tileexamples_n4_fwd_bwd, x, G, iters=100, backward=True
            )
            t_proj_n4 = benchmark(proj_n4_fwd_bwd, x, G, iters=100, backward=True)
            t_tl_proj_n4 = benchmark(tl_proj_n4_fwd_bwd, x, G, iters=100, backward=True)

            results.append({
                "N": N,
                "n": n,
                "Vanilla": t_vanilla / t_proj_n4,
                "Fused": t_fused / t_proj_n4,
                "SK-n4": t_sk_n4 / t_proj_n4,
                "TL-TileKernels-n4": t_tl_tilekernels_n4 / t_proj_n4,
                "TL-TileExamples-n4": t_tl_tileexamples_n4 / t_proj_n4,
                "TL-Proj-n4": t_tl_proj_n4 / t_proj_n4,
                "Proj-n4": t_proj_n4 / t_proj_n4,
            })
    dat = pd.DataFrame(results)
    print("=" * 70)
    print("Forward Pass + Backward Pass")
    print("=" * 70)
    print("Mean Run Time")
    print("=" * 70)
    print(dat.groupby(["N", "n"], as_index=False).mean().to_string(index=False))
    print("=" * 70)
    print("Median Run Time")
    print("=" * 70)
    print(dat.groupby(["N", "n"], as_index=False).median().to_string(index=False))
