import torch
from torch import nn
from .projector import Projector


class Encoder(nn.Module):
	def __init__(
			self,
			c_channels: int,
			d_channels: int,
			inner_mul_size: int = 4,
	):
		super().__init__()

		self.proj = Projector(in_channels=c_channels, out_channels=d_channels, inner_mul_size=inner_mul_size)

	def forward(self, image_tensor):
		image_latent = self.proj(image_tensor)
		return image_latent
