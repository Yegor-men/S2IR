import torch
import torch.nn as nn


class FiLM(nn.Module):
	def __init__(
			self,
			time_dim: int,
			num_channels: int,
			inner_mul_size: int = 4
	):
		super().__init__()

		self.layer_norm = nn.LayerNorm(time_dim)
		self.group_norm = nn.GroupNorm(1, num_channels)

		self.mlp = nn.Sequential(
			nn.Linear(in_features=time_dim, out_features=inner_mul_size * num_channels),
			nn.SiLU(),
		)

		self.gamma = nn.Linear(in_features=inner_mul_size * num_channels, out_features=num_channels)
		self.beta = nn.Linear(in_features=inner_mul_size * num_channels, out_features=num_channels)

		nn.init.normal_(self.gamma.weight, mean=0.0, std=1e-3)
		nn.init.zeros_(self.gamma.bias)

		nn.init.normal_(self.beta.weight, mean=0.0, std=1e-3)
		nn.init.zeros_(self.beta.bias)

	def forward(self, image_tensor, time_vector):
		b, c, h, w = image_tensor.shape

		normalized_image = self.group_norm(image_tensor)
		normalized_time = self.layer_norm(time_vector)

		gb = self.mlp(normalized_time)
		gamma, beta = self.gamma(gb) + 1.0, self.beta(gb)

		gamma = gamma.view(b, c, 1, 1)
		beta = beta.view(b, c, 1, 1)

		return normalized_image * gamma + beta
