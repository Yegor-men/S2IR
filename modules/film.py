import torch
from torch import nn


class FiLM(nn.Module):
	def __init__(self, time_dim: int, out_dim: int):
		super().__init__()

		self.film = nn.Linear(time_dim, 2 * out_dim)

		nn.init.normal_(self.film.weight, mean=0.0, std=1e-3)
		nn.init.zeros_(self.film.bias)

	def forward(self, time_cond):
		gb = self.film(time_cond)
		gamma, beta = gb.chunk(2, dim=-1)
		gamma = 1.0 + gamma

		return gamma, beta
