import torch
from torch import nn
from .axial_attention import AxialAttention
from .cross_attention import CrossAttention
from .image_film import ImageFiLM


class VisualTransformerBlock(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			text_cond_dim: int,
			text_cond_len: int,
			time_cond_dim: int,
			cross_dropout: float = 0.0,
			axial_dropout: float = 0.0,
			ffn_dropout: float = 0.0,
	):
		super().__init__()
		self.d_channels = d_channels

		self.cross_norm = nn.GroupNorm(num_heads, d_channels)
		self.cross_film = ImageFiLM(d_channels, d_channels)
		self.cross_attn = CrossAttention(
			d_channels=d_channels,
			num_heads=num_heads,
			text_cond_dim=text_cond_dim,
			text_cond_len=text_cond_len,
			time_cond_dim=time_cond_dim,
			dropout=cross_dropout
		)
		self.cross_scalar = nn.Parameter(torch.zeros(d_channels))

		self.axial_norm = nn.GroupNorm(num_heads, d_channels)
		self.axial_film = ImageFiLM(d_channels, d_channels)
		self.axial_attn = AxialAttention(
			d_channels=d_channels,
			num_heads=num_heads,
			dropout=axial_dropout
		)
		self.axial_scalar = nn.Parameter(torch.zeros(d_channels))

		self.ffn_norm = nn.GroupNorm(1, d_channels)
		self.ffn_film = ImageFiLM(d_channels, d_channels)
		self.ffn = nn.Sequential(
			nn.Conv2d(d_channels, 4 * d_channels, 1),
			nn.SiLU(),
			nn.Dropout(ffn_dropout),
			nn.Conv2d(4 * d_channels, d_channels, 1)
		)
		self.ffn_scalar = nn.Parameter(torch.zeros(d_channels))

	def forward(self, image, text_cond, time_cond):
		b, d, h, w = image.shape

		cross_norm = self.cross_norm(image)
		cross_filmed = self.cross_film(cross_norm, time_cond)
		cross_out = self.cross_attn(cross_filmed, text_cond, time_cond)

		image = image + cross_out * self.cross_scalar.view(1, self.d_channels, 1, 1)

		axial_norm = self.axial_norm(image)
		axial_filmed = self.axial_film(axial_norm, time_cond)
		axial_out = self.axial_attn(axial_filmed)

		image = image + axial_out * self.axial_scalar.view(1, self.d_channels, 1, 1)

		ffn_norm = self.ffn_norm(image)
		ffn_filmed = self.ffn_film(ffn_norm, time_cond)
		ffn_out = self.ffn(ffn_filmed)

		image = image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

		return image
