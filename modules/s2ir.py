import torch
from torch import nn

from .visual_transformer import VisualTransformerBlock
from .time_embedding import ContinuousTimeEmbed
from .position_embedding import RelPosEmbed2D


class SIIR(nn.Module):
	def __init__(
			self,
			c_channels: int,
			d_channels: int,
			num_heads: int,
			num_blocks: int,
			time_freq: int,
			time_dim: int,
			pos_freq: int,
			pos_dim: int,
			text_cond_dim: int,
			cond_dropout: float = 0.0,
			axial_dropout: float = 0.0,
			ffn_dropout: float = 0.0,
	):
		super().__init__()

		self.image_encoder = nn.Conv2d(c_channels, d_channels, 1)
		self.time_embed = ContinuousTimeEmbed(time_dim=time_dim, num_frequencies=time_freq)
		self.pos_embed = RelPosEmbed2D(pos_dim=pos_dim, num_frequencies=pos_freq)

		self.blocks = nn.ModuleList([VisualTransformerBlock(
			d_channels=d_channels,
			num_heads=num_heads,
			time_dim=time_dim,
			text_cond_dim=text_cond_dim,
			pos_cond_dim=pos_dim,
			cond_dropout=cond_dropout,
			axial_dropout=axial_dropout,
			ffn_dropout=ffn_dropout,
		) for _ in range(num_blocks)
		])

		self.image_decoder = nn.Sequential(
			nn.Conv2d(d_channels, 2 * d_channels, 1),
			nn.SiLU(),
			nn.Conv2d(2 * d_channels, c_channels, 1),
		)

		nn.init.zeros_(self.image_decoder[-1].weight)

	# nn.init.zeros_(self.image_decoder[-1].bias)

	def forward(self, image_tensor, text_cond, alpha_bar):
		b, c, h, w = image_tensor.shape

		image_latent = self.image_encoder(image_tensor)
		time_tensor = self.time_embed(alpha_bar)

		pos_cond = self.pos_embed(b, h, w)

		for index, block in enumerate(self.blocks):
			image_latent = block(image_latent, text_cond, pos_cond, time_tensor)

		predicted_noise = self.image_decoder(image_latent)

		return predicted_noise
