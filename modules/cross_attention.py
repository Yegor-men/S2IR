import torch
from torch import nn


class CrossAttention(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
	):
		super().__init__()
		assert d_channels % num_heads == 0
		self.cross_mha = nn.MultiheadAttention(embed_dim=d_channels, num_heads=num_heads, batch_first=True)

	def forward(self, image, guidance):
		B, D, H, W = image.shape

		q = image.permute(0, 2, 3, 1).reshape(B * H, W, D)
		k = guidance.permute(0, 2, 3, 1).reshape(B * H, W, D)
		v = k

		attn_out, _ = self.cross_mha(q, k, v, need_weights=False)
		attn_out = attn_out.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

		return attn_out


class CrossAdd(nn.Module):
	def __init__(self, d_channels):
		super().__init__()
		self.guidance_proj = nn.Conv2d(d_channels, d_channels, kernel_size=1)

	def forward(self, image, guidance):
		return image + self.guidance_proj(guidance)
