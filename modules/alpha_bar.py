import torch
import math


def alpha_bar_cosine(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
	eps = 1e-8
	factor = (t + s) / (1.0 + s)
	alpha_bar = torch.cos(factor * math.pi / 2.0).clamp(min=eps, max=1 - eps) ** 2
	return alpha_bar
