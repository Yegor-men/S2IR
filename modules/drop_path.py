import torch
import torch.nn as nn

def drop_path(x: torch.Tensor, drop_prob: float, training: bool, generator: torch.Generator = None) -> torch.Tensor:
    """Per-sample DropPath with no in-place ops. Returns a new tensor."""
    if (not training) or drop_prob == 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    # shape = [B, 1, 1, 1, ...] so mask broadcasts over channels/spatial dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    if generator is None:
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
    else:
        rand = torch.rand(shape, dtype=x.dtype, device=x.device, generator=generator)
    binary_mask = (rand < keep_prob).to(x.dtype)  # float mask, 1.0 or 0.0
    # Multiply a fresh tensor (no in-place) and scale so E[out] == x
    out = x * (binary_mask / keep_prob)
    return out

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
