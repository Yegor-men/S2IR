import torch
from torch import nn


class Projector(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.net = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=4 * out_channels, kernel_size=1),
			nn.SiLU(),
			nn.Conv2d(in_channels=4 * out_channels, out_channels=out_channels, kernel_size=1)
		)

	def forward(self, x: torch.Tensor):
		projected = self.net(x)
		return projected
