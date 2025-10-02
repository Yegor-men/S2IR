import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			dropout: float = 0.0,
	):
		super().__init__()
		assert d_channels % num_heads == 0

		self.d_channels = d_channels

		self.mha = nn.MultiheadAttention(
			embed_dim=d_channels,
			num_heads=num_heads,
			batch_first=True,
			dropout=dropout,
		)

		self.scalar = nn.Parameter(torch.ones(d_channels))

	def forward(self, image, text_tokens):
		b, d, h, w = image.shape

		s = h * w
		Q = image.permute(0, 2, 3, 1).contiguous().view(b, s, d)  # [B, S, D]

		# MHA wants shapes: (B, seq_q, D), (B, seq_k, D), (B, seq_k, D)
		attn_out, _ = self.mha(Q, text_tokens, text_tokens, need_weights=False)  # [B, S, D]

		# reshape back to image grid [B, D, H, W]
		attn_out = attn_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

		scalar = self.scalar.view(1, self.d_channels, 1, 1)

		return attn_out * scalar
