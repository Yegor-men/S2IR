import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .visual_transformer import VisualTransformerBlock
from .time_embedding import ContinuousTimeEmbed
from .projector import Projector


class SIIR(nn.Module):
	def __init__(
			self,
			c_channels: int,
			d_channels: int,
			num_heads: int,
			num_blocks: int,
			t_dim: int,
			text_embed_dim: int,
			pos_embed_dim: int,
			dropout_p_axial: float = 0.0,
			dropout_p_cross: float = 0.0,
			dropout_p_ffw: float = 0.0,
	):
		super().__init__()
		self.enc = Encoder(c_channels, d_channels)
		self.dec = Decoder(c_channels, d_channels)

		self.text_expander = Projector(in_channels=text_embed_dim, out_channels=d_channels)
		self.pos_expander = Projector(in_channels=pos_embed_dim, out_channels=d_channels)

		self.blocks = nn.ModuleList([VisualTransformerBlock(
			d_channels=d_channels,
			num_heads=num_heads,
			t_dim=t_dim,
			dropout_p_axial=dropout_p_axial,
			dropout_p_cross=dropout_p_cross,
			dropout_p_ffw=dropout_p_ffw,
		) for _ in range(num_blocks)
		])

		self.time_embed = ContinuousTimeEmbed(t_dim=t_dim, num_frequencies=t_dim // 2)

	def forward(self, image_tensor, text_cond, pos_cond, alpha_bar):
		time_vector = self.time_embed(alpha_bar)

		text_projection = self.text_expander(text_cond)
		pos_projection = self.pos_expander(pos_cond)
		guidance = text_projection + pos_projection

		image_latent = self.enc(image_tensor)

		for index, block in enumerate(self.blocks):
			image_latent = block(image_latent, guidance, time_vector)

		predicted_noise = self.dec(image_latent)

		return predicted_noise
