import torch
from torch import nn
from .film import FiLM
from .axial_attention import AxialAttention
from .cross_attention import CrossAttention, CrossAdd
from .projector import Projector


class VisualTransformerBlock(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			t_dim: int,
			dropout_p_axial: float = 0.0,
			dropout_p_cross: float = 0.0,
			dropout_p_ffw: float = 0.0,
	):
		super().__init__()
		self.norm1 = nn.GroupNorm(1, d_channels)
		self.film1 = FiLM(t_dim=t_dim, d_channels=d_channels)
		self.axial_norm = nn.GroupNorm(num_heads, d_channels)
		self.axial_attention = AxialAttention(d_channels=d_channels, num_heads=num_heads)
		self.axial_dropout = nn.Dropout(dropout_p_axial)
		self.axial_scalar = nn.Parameter(torch.ones(d_channels) * 1e-4)

		self.norm2 = nn.GroupNorm(1, d_channels)
		self.film2 = FiLM(t_dim=t_dim, d_channels=d_channels)
		self.cross_norm = nn.GroupNorm(num_heads, d_channels)
		self.cross_attention = CrossAttention(d_channels=d_channels, num_heads=num_heads)
		self.cross_dropout = nn.Dropout(dropout_p_cross)
		self.cross_scalar = nn.Parameter(torch.ones(d_channels) * 1e-4)

		self.norm3 = nn.GroupNorm(1, d_channels)
		self.film3 = FiLM(t_dim=t_dim, d_channels=d_channels)
		self.ffw = Projector(in_channels=d_channels, out_channels=d_channels, inner_mul_size=4, dropout=dropout_p_ffw)
		self.ffw_scalar = nn.Parameter(torch.ones(d_channels) * 1e-4)

	def forward(self, image_latent, guidance, time_vector):
		B, D, H, W = image_latent.shape

		x1_norm = self.norm1(image_latent)
		x1_mod = self.film1(x1_norm, time_vector)
		x1_mod_norm = self.axial_norm(x1_mod)
		axial_delta = self.axial_attention(x1_mod_norm)
		axial_delta = self.axial_dropout(axial_delta)
		axial_scale = self.axial_scalar.view(1, D, 1, 1)
		image_latent = image_latent + axial_delta * axial_scale

		x2_norm = self.norm2(image_latent)
		x2_mod = self.film2(x2_norm, time_vector)
		x2_mod_norm = self.cross_norm(x2_mod)
		cross_delta = self.cross_attention(x2_mod_norm, guidance)
		cross_delta = self.cross_dropout(cross_delta)
		cross_scale = self.cross_scalar.view(1, D, 1, 1)
		image_latent = image_latent + cross_delta * cross_scale

		x3_norm = self.norm3(image_latent)
		x3_mod = self.film3(x3_norm, time_vector)
		ffw_delta = self.ffw(x3_mod)
		ffw_scale = self.ffw_scalar.view(1, D, 1, 1)
		image_latent = image_latent + ffw_delta * ffw_scale

		return image_latent
