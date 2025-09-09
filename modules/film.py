import torch
import torch.nn as nn


class FiLM(nn.Module):
	def __init__(
			self,
			t_dim: int,
			d_channels: int,
			inner_mul_size: int = 4
	):
		super().__init__()

		self.mlp = nn.Sequential(
			nn.Linear(in_features=t_dim, out_features=inner_mul_size * d_channels),
			nn.SiLU(),
		)

		self.gamma = nn.Linear(in_features=inner_mul_size * d_channels, out_features=d_channels)
		self.beta = nn.Linear(in_features=inner_mul_size * d_channels, out_features=d_channels)

		nn.init.normal_(self.gamma.weight, mean=0.0, std=1e-3)
		nn.init.zeros_(self.gamma.bias)

		nn.init.normal_(self.beta.weight, mean=0.0, std=1e-3)
		nn.init.zeros_(self.beta.bias)

	def forward(self, image_tensor: torch.Tensor, time_vector: torch.Tensor) -> torch.Tensor:
		B, D, H, W = image_tensor.shape

		gb = self.mlp(time_vector)
		gamma, beta = self.gamma(gb) + 1.0, self.beta(gb)

		gamma = gamma.view(B, D, 1, 1)
		beta = beta.view(B, D, 1, 1)

		return image_tensor * gamma + beta
