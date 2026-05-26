import torch


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-image PSNR averaged across the batch. Inputs in [0, max_val]."""
    pred = pred.clamp(0.0, max_val)
    target = target.clamp(0.0, max_val)
    mse = ((pred - target) ** 2).mean(dim=(-3, -2, -1))
    mse = torch.clamp(mse, min=1e-10)
    return (10.0 * torch.log10(max_val ** 2 / mse)).mean()
