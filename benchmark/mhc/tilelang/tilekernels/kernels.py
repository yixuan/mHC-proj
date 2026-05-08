import torch

from .sinkhorn_kernel import _mhc_sinkhorn_bwd, _mhc_sinkhorn_fwd


_N4 = 4


def _check_n4_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3 or tensor.shape[-2:] != (_N4, _N4):
        raise ValueError(f"{name} must be a tensor of size B x 4 x 4")


def _cuda_float_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        tensor = tensor.to("cuda")
    return tensor.to(torch.float32).contiguous()


def sinkhorn_knopp_tilekernels_n4_forward(
    R: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 16,
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("R", R)
    src_options = {"device": R.device, "dtype": R.dtype}
    R_work = _cuda_float_contiguous(R)
    T_out = torch.empty_like(R_work)

    kernel = _mhc_sinkhorn_fwd(
        hidden_size=_N4,
        token_block_size=token_block_size,
        repeat=max_iter,
        eps=eps,
    )
    kernel(R_work, T_out)

    return {"T": T_out.to(**src_options)}


def sinkhorn_knopp_tilekernels_n4_backward(
    G: torch.Tensor,
    R: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 16,
) -> dict[str, torch.Tensor]:
    _check_n4_tensor("G", G)
    _check_n4_tensor("R", R)
    if G.shape != R.shape:
        raise ValueError("G and R must have the same shape")

    src_options = {"device": G.device, "dtype": G.dtype}
    G_work = _cuda_float_contiguous(G)
    R_work = _cuda_float_contiguous(R)
    D = torch.empty_like(G_work)

    kernel = _mhc_sinkhorn_bwd(
        hidden_size=_N4,
        token_block_size=token_block_size,
        repeat=max_iter,
        eps=eps,
    )
    kernel(G_work, R_work, D)

    return {"D": D.to(**src_options)}
