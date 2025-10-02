import torch
from torch import nn
import random

from .visual_transformer import VisualTransformerBlock
from .time_embedding import ContinuousTimeEmbed
from .position_embedding import RelPosEmbed2D
from .cross_attention import CrossAttention


class SIIR(nn.Module):
	def __init__(
			self,
			c_channels: int,
			d_channels: int,
			time_freq: int,
			time_dim: int,
			pos_freq: int,
			enc_blocks: int,
			dec_blocks: int,
			num_heads: int,
			text_cond_dim: int,
			text_token_length: int = 1,
			cross_dropout: float = 0.0,
			axial_dropout: float = 0.0,
			ffn_dropout: float = 0.0,
			pr_block_dropout: float = 0.0,
	):
		super().__init__()

		self.text_token_length = text_token_length
		self.d_channels = d_channels

		self.pos_embed = RelPosEmbed2D(num_frequencies=pos_freq, d_channels=d_channels)
		self.time_embed = ContinuousTimeEmbed(num_frequencies=time_freq, time_dim=time_dim)

		self.col_hint = nn.Conv2d(c_channels, d_channels, 1)
		self.pos_hint = nn.Sequential(
			nn.GroupNorm(1, 4 * pos_freq),
			nn.Conv2d(4 * pos_freq, d_channels, 1),
			nn.GroupNorm(1, d_channels),
		)

		self.enc_blocks = nn.ModuleList([
			VisualTransformerBlock(
				d_channels=d_channels,
				num_heads=num_heads,
				time_dim=time_dim,
				axial_dropout=axial_dropout,
				ffn_dropout=ffn_dropout,
				block_dropout=pr_block_dropout,
			) for _ in range(enc_blocks)
		])

		self.semantic_norm = nn.GroupNorm(1, d_channels)
		self.col_hint2 = nn.Conv2d(c_channels, d_channels, 1)
		self.pos_hint2 = nn.Sequential(
			nn.GroupNorm(1, 4 * pos_freq),
			nn.Conv2d(4 * pos_freq, d_channels, 1),
			# nn.GroupNorm(1, d_channels),
		)

		self.text_proj = nn.Linear(in_features=text_cond_dim, out_features=text_token_length * d_channels)
		self.token_norm = nn.LayerNorm(d_channels)

		self.cross_attn = CrossAttention(
			d_channels=d_channels,
			num_heads=1,
			dropout=cross_dropout,
		)

		self.dec_blocks = nn.ModuleList([
			VisualTransformerBlock(
				d_channels=d_channels,
				num_heads=num_heads,
				time_dim=time_dim,
				axial_dropout=axial_dropout,
				ffn_dropout=ffn_dropout,
				block_dropout=pr_block_dropout,
			) for _ in range(dec_blocks)
		])

		self.image_decoder = nn.Conv2d(d_channels, c_channels, 1)
		nn.init.zeros_(self.image_decoder.weight)
		nn.init.zeros_(self.image_decoder.bias)

	def forward(
			self,
			noisy_image,
			alpha_bar,
			pos_cond=None,
			neg_cond=None,
	):
		b, c, h, w = noisy_image.shape

		pos_freqs = self.pos_embed(h, w).unsqueeze(0)
		time_tensor = self.time_embed(alpha_bar)  # [B, T]

		col_hint1 = self.col_hint(noisy_image)  # [b, d, h, w]
		pos_hint1 = self.pos_hint(pos_freqs)  # [1, d, h, w]

		semantic_map = col_hint1 + pos_hint1

		for i, block in enumerate(self.enc_blocks):
			semantic_map = block(semantic_map, time_tensor)

		col_hint2 = self.col_hint2(noisy_image)
		pos_hint2 = self.pos_hint2(pos_freqs)

		semantic_map = self.semantic_norm(semantic_map) + col_hint2 + pos_hint2

		# EPS NULL
		eps_null = semantic_map
		for i, block in enumerate(self.dec_blocks):
			eps_null = block(eps_null, time_tensor)
		eps_null = self.image_decoder(eps_null)

		# EPS POS
		if pos_cond is not None:
			eps_pos = semantic_map
			pos_tokens = self.text_proj(pos_cond).view(b, self.text_token_length, self.d_channels)  # [B, L, D]
			pos_tokens = self.token_norm(pos_tokens)
			pos_cross_delta = self.cross_attn(eps_pos, pos_tokens)
			eps_pos = eps_pos + pos_cross_delta
			for i, block in enumerate(self.dec_blocks):
				eps_pos = block(eps_pos, time_tensor)
			eps_pos = self.image_decoder(eps_pos)
		else:
			eps_pos = None

		# EPS NEG
		if neg_cond is not None:
			eps_neg = semantic_map
			neg_tokens = self.text_proj(neg_cond).view(b, self.text_token_length, self.d_channels)  # [B, L, D]
			neg_tokens = self.token_norm(neg_tokens)
			neg_cross_delta = self.cross_attn(eps_neg, neg_tokens)
			eps_neg = eps_neg + neg_cross_delta
			for i, block in enumerate(self.dec_blocks):
				eps_neg = block(eps_neg, time_tensor)
			eps_neg = self.image_decoder(eps_neg)
		else:
			eps_neg = None

		return eps_null, eps_pos, eps_neg
