import torch
import torch.nn as nn
from . import _internal

class MHCProjectionN4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, tol=1e-6):
        in_shape = R.shape
        R = R.view(-1, 4, 4)
        res = _internal.torch.birkhoff_proj_n4(R, tol)
        T = res["T"]

        ctx.save_for_backward(T)
        ctx.set_materialize_grads(False)

        return T.view(in_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Early exit if R does not require gradient
        if not ctx.needs_input_grad[0]:
            return None, None

        in_shape = grad_output.shape
        G = grad_output.view(-1, 4, 4)
        T = ctx.saved_tensors[0]
        res = _internal.torch.birkhoff_proj_n4_backward(G, T)
        D = res["D"]

        return D.view(in_shape), None

class MHCProjectionN4(nn.Module):
    def __init__(self, tol=1e-6):
        super().__init__()
        self.tol = tol

    def forward(self, x):
        in_shape = x.shape
        if x.ndim < 2 or in_shape[-1] != 4 or in_shape[-2] != 4:
            raise ValueError("x must be a tensor of size B x 4 x 4")
        out = MHCProjectionN4Function.apply(x, self.tol)
        return out



class MHCSinkhornN4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, max_iter=20):
        in_shape = R.shape
        expon = torch.exp(R.view(-1, 4, 4))
        res = _internal.torch.sinkhorn_knopp_n4(expon, max_iter)
        T = res["T"]

        ctx.max_iter = max_iter
        ctx.save_for_backward(expon)
        ctx.set_materialize_grads(False)

        return T.view(in_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Early exit if R does not require gradient
        if not ctx.needs_input_grad[0]:
            return None, None

        in_shape = grad_output.shape
        G = grad_output.view(-1, 4, 4)
        expon = ctx.saved_tensors[0]
        res = _internal.torch.sinkhorn_knopp_n4_backward(G, expon, ctx.max_iter)
        D = res["D"] * expon

        return D.view(in_shape), None

class MHCSinkhornN4(nn.Module):
    def __init__(self, max_iter=20):
        super().__init__()
        self.max_iter = max_iter

    def forward(self, x):
        in_shape = x.shape
        if x.ndim < 2 or in_shape[-1] != 4 or in_shape[-2] != 4:
            raise ValueError("x must be a tensor of size B x 4 x 4")
        out = MHCSinkhornN4Function.apply(x, self.max_iter)
        return out
