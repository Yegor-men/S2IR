import torch
from torch import nn

from .visual_transformer import VisualTransformerBlock
from .time_embedding import ContinuousTimeEmbed
from .position_embedding import RelPosEmbed2D
from .color_embed import ColorEmbedding


class SIIR(nn.Module):
	def __init__(
			self,
			c_channels: int,
			d_channels: int,
			col_freq: int,
			time_freq: int,
			time_dim: int,
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

		self.text_proj = nn.Sequential(
			nn.Linear(in_features=text_cond_dim, out_features=2 * text_token_length * d_channels),
			nn.SiLU(),
			nn.Linear(in_features=2 * text_token_length * d_channels, out_features=text_token_length * d_channels)
		)
		self.token_norm = nn.LayerNorm(d_channels)

		self.time_embed = ContinuousTimeEmbed(num_frequencies=time_freq, time_dim=time_dim)

		self.pos_embed = RelPosEmbed2D(num_frequencies=pos_freq, d_channels=d_channels)

		self.blocks = nn.ModuleList([VisualTransformerBlock(
			d_channels=d_channels,
			num_heads=num_heads,
			time_dim=time_dim,
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

		image_latent = self.image_encoder(image_tensor)  # [B, D, H, W], unnormalized
		pos_cond = self.pos_embed(h, w)  # [D, H, W], normalized
		tokens = self.text_proj(text_cond).view(b, self.text_token_length, self.d_channels)  # [B, L, D]
		text_cond = self.token_norm(tokens)
		time_tensor = self.time_embed(alpha_bar)  # [B, T]

		for index, block in enumerate(self.blocks):
			image_latent = block(
				image_latent,
				pos_cond,
				text_cond,
				time_tensor
			)

		predicted_noise = self.image_decoder(image_latent)

		return predicted_noise
