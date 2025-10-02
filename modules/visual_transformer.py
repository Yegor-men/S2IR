import torch
from torch import nn
from .axial_attention import AxialAttention
from .cross_attention import CrossAttention
from .film import FiLM
from .drop_path import DropPath


class VisualTransformerBlock(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			time_dim: int,
			axial_dropout: float = 0.0,
			ffn_dropout: float = 0.0,
			block_dropout: float = 0.0,
	):
		super().__init__()
		self.d_channels = d_channels

		self.drop_path = DropPath(block_dropout)

		self.axial_norm = nn.GroupNorm(num_heads, d_channels)
		self.axial_film = FiLM(time_dim, d_channels)
		self.axial_attn = AxialAttention(
			d_channels=d_channels,
			num_heads=num_heads,
			dropout=axial_dropout
		)
		self.axial_scalar = nn.Parameter(torch.ones(d_channels))

		self.ffn_norm = nn.GroupNorm(1, d_channels)
		self.ffn_film = FiLM(time_dim, d_channels)
		# self.ffn = nn.Sequential(
		# 	nn.Conv2d(d_channels, 2 * d_channels, 1),
		# 	nn.SiLU(),
		# 	nn.Dropout(ffn_dropout),
		# 	nn.Conv2d(2 * d_channels, 2 * d_channels, 1),
		# 	nn.SiLU(),
		# 	nn.Dropout(ffn_dropout),
		# 	nn.Conv2d(2 * d_channels, d_channels, 1)
		# )
		self.ffn = nn.Sequential(
			nn.Conv2d(d_channels, 4 * d_channels, 1),
			nn.SiLU(),
			nn.Dropout(ffn_dropout),
			nn.Conv2d(4 * d_channels, d_channels, 1)
		)
		self.ffn_scalar = nn.Parameter(torch.ones(d_channels))

		self.final_scalar = nn.Parameter(torch.ones(d_channels) * 1e-3)
		self.final_film = FiLM(time_dim, d_channels)

	def forward(self, image, time_cond):
		b, d, h, w = image.shape

		working_image = image

		axial_norm = self.axial_norm(working_image)
		axial_g, axial_b = self.axial_film(time_cond)
		axial_filmed = axial_norm * axial_g.unsqueeze(-1).unsqueeze(-1) + axial_b.unsqueeze(-1).unsqueeze(-1)
		axial_out = self.axial_attn(axial_filmed)

		working_image = working_image + axial_out * self.axial_scalar.view(1, self.d_channels, 1, 1)

		ffn_norm = self.ffn_norm(working_image)
		ffn_g, ffn_b = self.ffn_film(time_cond)
		ffn_filmed = ffn_norm * ffn_g.unsqueeze(-1).unsqueeze(-1) + ffn_b.unsqueeze(-1).unsqueeze(-1)
		ffn_out = self.ffn(ffn_filmed)

		working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

		blend_scalar = self.final_scalar.view(1, self.d_channels, 1, 1)
		final_g, final_b = self.final_film(time_cond)
		blend_scalar = final_g.unsqueeze(-1).unsqueeze(-1) * blend_scalar + final_b.unsqueeze(-1).unsqueeze(-1)
		final_image = image + blend_scalar * self.drop_path(working_image)

		return final_image
