import torch
import math


def alpha_bar_cosine(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
	factor = (t + s) / (1.0 + s)
	alpha_bar = torch.cos(factor * math.pi / 2.0).clamp(min=0.0, max=1.0) ** 2
	return alpha_bar
