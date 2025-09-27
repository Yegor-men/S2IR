import torch
from torch import nn
from .axial_attention import AxialAttention
from .cross_attention import CrossAttention
from .film import FiLM


class VisualTransformerBlock(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			time_dim: int,
			cross_dropout: float = 0.0,
			axial_dropout: float = 0.0,
			ffn_dropout: float = 0.0,
	):
		super().__init__()
		self.d_channels = d_channels

		self.cross_norm = nn.GroupNorm(1, d_channels)
		self.cross_film = FiLM(time_dim, d_channels)
		self.cross_attn = CrossAttention(
			d_channels=d_channels,
			num_heads=1,
			dropout=cross_dropout
		)
		self.cross_scalar = nn.Parameter(torch.ones(d_channels) * 1e-4)

		self.axial_norm = nn.GroupNorm(num_heads, d_channels)
		self.axial_film = FiLM(time_dim, d_channels)
		self.axial_attn = AxialAttention(
			d_channels=d_channels,
			num_heads=num_heads,
			dropout=axial_dropout
		)
		self.axial_scalar = nn.Parameter(torch.ones(d_channels) * 1e-4)

		self.ffn_norm = nn.GroupNorm(1, d_channels)
		self.ffn_film = FiLM(time_dim, d_channels)
		self.ffn = nn.Sequential(
			nn.Conv2d(d_channels, 2 * d_channels, 1),
			nn.SiLU(),
			nn.Dropout(ffn_dropout),
			nn.Conv2d(2 * d_channels, 2 * d_channels, 1),
			nn.SiLU(),
			nn.Dropout(ffn_dropout),
			nn.Conv2d(2 * d_channels, d_channels, 1)
		)
		self.ffn_scalar = nn.Parameter(torch.ones(d_channels) * 1e-4)

		self.final_scalar = nn.Parameter(torch.ones(d_channels) * 1e-3)
		self.final_film = FiLM(time_dim, d_channels)

	def forward(self, image, pos_cond, text_cond, time_cond):
		b, d, h, w = image.shape

		working_image = image + pos_cond

		cross_norm = self.cross_norm(working_image)
		cross_g, cross_b = self.cross_film(time_cond)
		cross_filmed = cross_norm * cross_g.unsqueeze(-1).unsqueeze(-1) + cross_b.unsqueeze(-1).unsqueeze(-1)
		cross_out = self.cross_attn(cross_filmed, text_cond)

		working_image = working_image + cross_out * self.cross_scalar.view(1, self.d_channels, 1, 1)

		axial_norm = self.axial_norm(working_image)
		axial_g, axial_b = self.cross_film(time_cond)
		axial_filmed = axial_norm * axial_g.unsqueeze(-1).unsqueeze(-1) + axial_b.unsqueeze(-1).unsqueeze(-1)
		axial_out = self.axial_attn(axial_filmed)

		working_image = working_image + axial_out * self.axial_scalar.view(1, self.d_channels, 1, 1)

		ffn_norm = self.ffn_norm(working_image)
		ffn_g, ffn_b = self.cross_film(time_cond)
		ffn_filmed = ffn_norm * ffn_g.unsqueeze(-1).unsqueeze(-1) + ffn_b.unsqueeze(-1).unsqueeze(-1)
		ffn_out = self.ffn(ffn_filmed)

		working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

		# blend_scalar = nn.functional.sigmoid(self.final_scalar).view(1, self.d_channels, 1, 1)
		# final_image = image * (1 - blend_scalar) + working_image * blend_scalar

		blend_scalar = self.final_scalar.view(1, self.d_channels, 1, 1)
		final_g, final_b = self.final_film(time_cond)
		blend_scalar = final_g.unsqueeze(-1).unsqueeze(-1) * blend_scalar + final_b.unsqueeze(-1).unsqueeze(-1)
		final_image = image + working_image * blend_scalar

		return final_image
