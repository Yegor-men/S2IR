import torch
from torch import nn


class Projector(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			inner_mul_size: int = 4,
			zero_last_layer: bool = False,
	):
		super().__init__()

		self.net = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=inner_mul_size * out_channels, kernel_size=1),
			nn.SiLU(),
			nn.Conv2d(in_channels=inner_mul_size * out_channels, out_channels=out_channels, kernel_size=1)
		)

		if zero_last_layer:
			last_conv = self.net[-1]
			nn.init.normal_(last_conv.weight, mean=0.0, std=1e-3)
			nn.init.zeros_(last_conv.bias)

	def forward(self, x: torch.Tensor):
		projected = self.net(x)
		return projected
