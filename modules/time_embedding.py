import torch
from torch import nn
import math


class ContinuousTimeEmbed(nn.Module):
	def __init__(
			self,
			num_frequencies: int,
			time_dim: int,
			eps: float = 1e-6
	):
		super().__init__()

		self.eps = float(eps)

		powers = torch.arange(num_frequencies, dtype=torch.float32)
		frequencies = math.pi * (2.0 ** powers)  # [pi, 2pi, 4pi, ...]
		self.register_buffer("frequencies", frequencies, persistent=True)

		self.proj = nn.Sequential(
			nn.LayerNorm(2 * num_frequencies),
			nn.Linear(2 * num_frequencies, time_dim),
			nn.SiLU()
		)

	def forward(self, alpha_bar: torch.Tensor) -> torch.Tensor:
		alpha_mapped = alpha_bar * (1 - 2 * self.eps) - (0.5 - self.eps)
		# Now it's between [-0.5 + eps, 0.5 - eps]

		tproj = alpha_mapped.unsqueeze(1) * self.frequencies.view(1, -1)
		sin_feat = torch.sin(tproj)
		cos_feat = torch.cos(tproj)
		feat = torch.cat([sin_feat, cos_feat], dim=-1)

		time_tensor = self.proj(feat)

		return time_tensor
