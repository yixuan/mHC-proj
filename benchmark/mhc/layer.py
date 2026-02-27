import torch
import torch.nn as nn
import triton

from .kernels import _mhc_sinkhorn_fwd_kernel, _mhc_sinkhorn_bwd_kernel


class MHCSinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, num_iters=5):
        W = W.contiguous()
        B, n, _ = W.shape
        M = torch.empty_like(W)

        NN = n * n
        BLOCK_SIZE = triton.next_power_of_2(NN)

        # History: [B, ITERS*2*NN] fp32 (row-phase + col-phase per iter)
        H = torch.empty((B, num_iters * 2 * NN), device=W.device, dtype=torch.float32)

        _mhc_sinkhorn_fwd_kernel[(B,)](
            W, M, H,
            W.stride(0), H.stride(0),
            N_LANES=n,
            ITERS=num_iters,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(W, H)
        ctx.n_lanes = n
        ctx.num_iters = num_iters
        return M

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        W, H = ctx.saved_tensors

        grad_W = torch.empty_like(W)
        B, n, iters = W.shape[0], ctx.n_lanes, ctx.num_iters

        NN = n * n
        BLOCK_SIZE = triton.next_power_of_2(NN)

        _mhc_sinkhorn_bwd_kernel[(B,)](
            grad_output, W, H, grad_W,
            W.stride(0), H.stride(0),
            N_LANES=n,
            ITERS=iters,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_W, None


class FusedMHC(nn.Module):
    def __init__(self, mhc_iters=5):
        super().__init__()
        self.mhc_iters = mhc_iters

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        input_dtype = x.dtype
        input_shape = x.shape
        n_lanes = input_shape[-1]

        x_flat = x.contiguous().view(-1, n_lanes, n_lanes).float()
        out_flat = MHCSinkhornFunction.apply(x_flat, self.mhc_iters)
        return out_flat.view(input_shape).to(input_dtype)


def mhc_warmup(n_lanes=4, batch_size=32, device="cuda"):
    if not torch.cuda.is_available():
        return

    print(f"🔥 Warming up MHC kernels for {n_lanes} lanes...")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dummy = torch.randn(batch_size, n_lanes, n_lanes, device=device, requires_grad=True)
        layer = FusedMHC(mhc_iters=5).to(device)
        out = layer(dummy)
        out.sum().backward()

    torch.cuda.current_stream().wait_stream(stream)
    print("✅ MHC Warmup Complete.")
