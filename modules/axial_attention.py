import torch
from torch import nn


class AxialAttention(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			dropout: float = 0.0
	):
		super().__init__()
		assert d_channels % num_heads == 0

		self.row_mha = nn.MultiheadAttention(
			embed_dim=d_channels,
			num_heads=num_heads,
			batch_first=True,
			dropout=dropout,
		)
		self.col_mha = nn.MultiheadAttention(
			embed_dim=d_channels,
			num_heads=num_heads,
			batch_first=True,
			dropout=dropout,
		)

	def forward(self, image):
		B, D, H, W = image.shape

		x_row = image.permute(0, 2, 3, 1).contiguous().view(B * H, W, D)
		attn_row_out, _ = self.row_mha(x_row, x_row, x_row, need_weights=False)
		attn_row_out = attn_row_out.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

		x_col = image.permute(0, 3, 2, 1).contiguous().view(B * W, H, D)
		attn_col_out, _ = self.col_mha(x_col, x_col, x_col, need_weights=False)
		attn_col_out = attn_col_out.view(B, W, H, D).permute(0, 3, 2, 1).contiguous()

		return attn_row_out + attn_col_out
