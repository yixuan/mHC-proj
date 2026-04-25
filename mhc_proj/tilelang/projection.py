import torch


def birkhoff_proj_n4_forward(
    R: torch.Tensor, tol: float = 1e-6
) -> dict[str, torch.Tensor]: ...


def birkhoff_proj_n4_backward(
    G: torch.Tensor, T: torch.Tensor
) -> dict[str, torch.Tensor]: ...


def sinkhorn_knopp_n4_forward(
    R: torch.Tensor, max_iter: int = 20
) -> dict[str, torch.Tensor]: ...


def sinkhorn_knopp_n4_backward(
    G: torch.Tensor, expM: torch.Tensor, max_iter: int = 20
) -> dict[str, torch.Tensor]: ...
