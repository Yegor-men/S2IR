import torch
from torch import nn
from .projector import Projector


class Decoder(nn.Module):
	def __init__(
			self,
			c_channels: int,
			d_channels: int,
			inner_mul_size: int = 4,
	):
		super().__init__()

		self.proj = Projector(
			in_channels=d_channels,
			out_channels=c_channels,
			inner_mul_size=inner_mul_size,
			zero_last_layer=True
		)

	def forward(self, image_latent):
		image_tensor = self.proj(image_latent)
		return image_tensor
