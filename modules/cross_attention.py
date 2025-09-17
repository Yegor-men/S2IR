import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
	def __init__(
			self,
			d_channels: int,
			num_heads: int,
			text_cond_dim: int,
			text_cond_len: int,
			time_cond_dim: int,
			dropout: float = 0.0,
	):
		super().__init__()
		assert d_channels % num_heads == 0

		self.text_token_length = int(text_cond_len)
		self.d_channels = int(d_channels)
		self.num_heads = int(num_heads)

		self.text_proj = nn.Linear(in_features=text_cond_dim, out_features=self.text_token_length * d_channels)

		self.text_film = nn.Sequential(
			nn.Linear(time_cond_dim, 4 * self.text_token_length * d_channels),
			nn.SiLU(),
			nn.Linear(4 * self.text_token_length * d_channels, 2 * self.text_token_length * d_channels)
		)
		nn.init.normal_(self.text_film[-1].weight, mean=0.0, std=1e-3)
		nn.init.zeros_(self.text_film[-1].bias)

		self.token_norm = nn.LayerNorm(d_channels)

		self.mha = nn.MultiheadAttention(
			embed_dim=d_channels,
			num_heads=num_heads,
			batch_first=True,
			dropout=dropout,
		)

	def forward(self, image, text_cond, time_cond):
		b, d, h, w = image.shape

		# ---- create L text tokens of dim D ----
		# project text (E -> L*D)
		proj_text = self.text_proj(text_cond)  # [B, L*D]
		# FiLM from time embedding -> 2 * L * D
		time_gb = self.text_film(time_cond)  # [B, 2*L*D]
		gamma, beta = time_gb.chunk(2, dim=-1)  # each [B, L*D]

		# apply FiLM to projected text tokens
		proj_text = proj_text * (gamma + 1.0) + beta  # [B, L*D]

		# reshape into tokens: [B, L, D]
		text_tokens = proj_text.view(b, self.text_token_length, self.d_channels)

		# normalize tokens along D
		text_tokens = self.token_norm(text_tokens)  # LayerNorm over last dim

		# ---- form queries from image pixels ----
		# image is already normalized upstream (you said), but we still need to form Q as [B, S, D]
		s = h * w
		Q = image.permute(0, 2, 3, 1).contiguous().view(b, s, d)  # [B, S, D]

		# optional: layer-norm Q (not mandatory if you already normalized image)
		# Q = self.token_norm(Q)

		# ---- run multihead attention: queries = pixels, keys/values = text tokens ----
		# MHA wants shapes: (B, seq_q, D), (B, seq_k, D), (B, seq_k, D)
		attn_out, _ = self.mha(Q, text_tokens, text_tokens, need_weights=False)  # [B, S, D]

		# reshape back to image grid [B, D, H, W]
		attn_out = attn_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

		return attn_out
