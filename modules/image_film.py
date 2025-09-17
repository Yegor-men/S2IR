import torch
from torch import nn


class ImageFiLM(nn.Module):
	def __init__(self, d_channels: int, time_cond_dim: int):
		super().__init__()

		self.film = nn.Sequential(
			nn.Linear(in_features=time_cond_dim, out_features=4 * d_channels),
			nn.SiLU(),
			nn.Linear(in_features=4 * d_channels, out_features=2 * d_channels),
		)
		nn.init.normal_(self.film[-1].weight, mean=0.0, std=1e-3)
		nn.init.zeros_(self.film[-1].bias)

	def forward(self, image, time_cond):
		gb = self.film(time_cond)
		gamma, beta = gb.chunk(2, dim=-1)
		gamma = (1.0 + gamma).unsqueeze(-1).unsqueeze(-1)
		beta = beta.unsqueeze(-1).unsqueeze(-1)

		return image * gamma + beta
