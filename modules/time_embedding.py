import torch
from torch import nn
import math


class ContinuousTimeEmbed(nn.Module):
	def __init__(
			self,
			t_dim: int = 128,
			num_frequencies: int = 64,
			max_period: float = 10000.0
	):
		super().__init__()

		frequencies = torch.exp(
			torch.linspace(0, math.log(max_period), num_frequencies)
		)
		self.register_buffer("frequencies", frequencies * 2 * math.pi, persistent=False)

		self.mlp = nn.Sequential(
			nn.Linear(num_frequencies * 2, t_dim),
			nn.SiLU(),
			nn.Linear(t_dim, t_dim),
		)

	def forward(self, alpha_bar: torch.Tensor) -> torch.Tensor:
		B = alpha_bar.shape[0]

		t_proj = alpha_bar.unsqueeze(1) * self.frequencies
		sin_feat = torch.sin(t_proj)
		cos_feat = torch.cos(t_proj)
		feat = torch.cat([sin_feat, cos_feat], dim=-1)

		t_emb = self.mlp(feat)

		return t_emb
