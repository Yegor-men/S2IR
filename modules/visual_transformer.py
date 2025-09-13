import torch
from torch import nn
from .film import FiLM
from .axial_attention import AxialAttention


class VisualTransformerBlock(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			time_dim: int,
			text_cond_dim: int,
			pos_cond_dim: int,
			cond_dropout: float = 0.0,
			axial_dropout: float = 0.0,
			ffn_dropout: float = 0.0,
	):
		super().__init__()
		self.d_channels = d_channels

		self.image_norm = nn.GroupNorm(1, d_channels)

		self.text_film = FiLM(time_dim=time_dim, num_channels=text_cond_dim)
		self.cross_add_text = nn.Conv2d(text_cond_dim, d_channels, 1)
		self.text_dropout = nn.Dropout2d(cond_dropout)

		self.pos_film = FiLM(time_dim=time_dim, num_channels=pos_cond_dim)
		self.cross_add_pos = nn.Conv2d(pos_cond_dim, d_channels, 1)
		self.pos_dropout = nn.Dropout2d(cond_dropout)

		self.axial_film = FiLM(time_dim=time_dim, num_channels=d_channels)
		self.axial_attn = AxialAttention(d_channels=d_channels, num_heads=num_heads, dropout=axial_dropout)
		self.axial_scalar = nn.Parameter(torch.full((d_channels,), -6.0))

		self.ffn_film = FiLM(time_dim=time_dim, num_channels=d_channels)
		self.ffn = nn.Sequential(
			nn.Conv2d(d_channels, 4 * d_channels, 1),
			nn.SiLU(),
			nn.Dropout(ffn_dropout),
			nn.Conv2d(4 * d_channels, d_channels, 1),
		)
		self.ffn_scalar = nn.Parameter(torch.full((d_channels,), -6.0))

		self.final_scalar = nn.Parameter(torch.full((d_channels,), -6.0))

	def forward(self, image_tensor, text_cond, pos_cond, time_tensor):
		normalized_image = self.image_norm(image_tensor)

		filmed_text = self.text_film(text_cond, time_tensor)
		filmed_pos = self.pos_film(pos_cond, time_tensor)
		text_conditioning = self.cross_add_text(filmed_text)
		pos_conditioning = self.cross_add_pos(filmed_pos)
		text_conditioning = self.text_dropout(text_conditioning)
		pos_conditioning = self.pos_dropout(pos_conditioning)
		conditioned_image = normalized_image + text_conditioning + pos_conditioning

		filmed_axial = self.axial_film(conditioned_image, time_tensor)
		axial_delta = self.axial_attn(filmed_axial)
		axial_scalar = nn.functional.sigmoid(self.axial_scalar).view(1, self.d_channels, 1, 1)
		attended_image = conditioned_image * (1 - axial_scalar) + axial_delta * axial_scalar

		filmed_ffn = self.ffn_film(attended_image, time_tensor)
		ffn_delta = self.ffn(filmed_ffn)
		ffn_scalar = nn.functional.sigmoid(self.ffn_scalar).view(1, self.d_channels, 1, 1)
		modified_image = attended_image * (1 - ffn_scalar) + ffn_delta * ffn_scalar

		final_scalar = nn.functional.sigmoid(self.final_scalar).view(1, self.d_channels, 1, 1)
		final_image = image_tensor * (1 - final_scalar) + modified_image * final_scalar

		return final_image
