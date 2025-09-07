import torch
import torch.nn as nn


class FiLM(nn.Module):
	def __init__(self, t_dim: int, d_channels: int):
		super().__init__()
		self.t_dim = t_dim
		self.d = d_channels

		self.gamma_beta = nn.Sequential(
			nn.Linear(in_features=t_dim, out_features=4 * d_channels),
			nn.SiLU(),
			nn.Linear(in_features=4 * d_channels, out_features=d_channels)
		)

		nn.init.zeros_(self.gamma_beta.bias)
		nn.init.normal_(self.gamma_beta.weight, mean=0.0, std=1e-3)

	def forward(self, image_tensor: torch.Tensor, time_vector: torch.Tensor) -> torch.Tensor:
		B, D, H, W = image_tensor.shape

		gb = self.gamma_beta(time_vector)
		gamma, beta = gb.chunk(2, dim=-1)
		gamma = gamma + 1.0

		gamma = gamma.view(B, D, 1, 1)
		beta = beta.view(B, D, 1, 1)

		return image_tensor * gamma + beta
