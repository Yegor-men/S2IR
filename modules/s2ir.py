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
			time_freq: int,
			pos_freq: int,
			num_blocks: int,
			num_heads: int,
			text_cond_dim: int,
			text_token_length: int = 1,
			cross_dropout: float = 0.0,
			axial_dropout: float = 0.0,
			ffn_dropout: float = 0.0,
	):
		super().__init__()

		self.text_token_length = text_token_length
		self.d_channels = d_channels

		self.image_encoder = nn.Conv2d(c_channels, d_channels, 1)
		self.text_proj = nn.Linear(in_features=text_cond_dim, out_features=text_token_length * d_channels)
		self.time_embed = ContinuousTimeEmbed(num_frequencies=time_freq)
		self.pos_embed = RelPosEmbed2D(num_frequencies=pos_freq, d_channels=d_channels)

		self.blocks = nn.ModuleList([VisualTransformerBlock(
			d_channels=d_channels,
			num_heads=num_heads,
			time_freq=time_freq,
			cross_dropout=cross_dropout,
			axial_dropout=axial_dropout,
			ffn_dropout=ffn_dropout,
		) for _ in range(num_blocks)
		])

		self.image_decoder = nn.Conv2d(d_channels, c_channels, 1)

		nn.init.zeros_(self.image_decoder.weight)
		nn.init.zeros_(self.image_decoder.bias)

	def forward(self, image_tensor, text_cond, alpha_bar):
		b, c, h, w = image_tensor.shape

		image_latent = self.image_encoder(image_tensor) + self.pos_embed(h, w)
		text_tokens = self.text_proj(text_cond).view(b, self.text_token_length, self.d_channels)  # [B, L, D]
		time_tensor = self.time_embed(alpha_bar)

		for index, block in enumerate(self.blocks):
			image_latent = block(image_latent, text_tokens, time_tensor)

		predicted_noise = self.image_decoder(image_latent)

		return predicted_noise
