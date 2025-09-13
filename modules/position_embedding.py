# rel_pos_embed2d.py
import math
import torch
from torch import nn


class RelPosEmbed2D(nn.Module):
	def __init__(
			self,
			pos_dim: int = 64,
			num_frequencies: int = 16,
			eps: float = 1e-6
	):
		super().__init__()
		self.pos_dim = int(pos_dim)
		self.num_frequencies = int(num_frequencies)
		self.eps = float(eps)

		base = 2.0 * math.pi
		powers = torch.arange(self.num_frequencies, dtype=torch.float32)
		freqs = base * (2.0 ** powers)
		self.register_buffer("frequencies", freqs, persistent=True)

		in_ch = 4 * self.num_frequencies
		hidden = max(self.pos_dim, in_ch)

		self.proj = nn.Sequential(
			nn.Conv2d(in_channels=in_ch, out_channels=hidden, kernel_size=1),
			nn.SiLU(),
			nn.Conv2d(in_channels=hidden, out_channels=self.pos_dim, kernel_size=1),
		)

	def _make_grid(self, h: int, w: int):
		if w >= h:
			x_min, x_max = -0.25, 0.25
			y_extent = h / w
			y_min, y_max = -0.25 * y_extent, 0.25 * y_extent
		else:
			y_min, y_max = -0.25, 0.25
			x_extent = w / h
			x_min, x_max = -0.25 * x_extent, 0.25 * x_extent

		x_coords = torch.linspace(x_min + self.eps, x_max - self.eps, steps=w)
		y_coords = torch.linspace(y_min + self.eps, y_max - self.eps, steps=h)

		yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
		grid = torch.stack([xx, yy], dim=0)
		return grid

	def forward(self, b: int, h: int, w: int) -> torch.Tensor:
		grid = self._make_grid(h, w).to(self.frequencies)  # [2, H, W]
		grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # [B, 2, H, W]

		# attach frequency axis [B, 2, H, W, 1] * [1, 1, 1, 1, F] -> [B,2,H,W,F]
		grid_unsq = grid.unsqueeze(-1)  # [B,2,H,W,1]
		freqs = self.frequencies.view(1, 1, 1, 1, -1)  # [1,1,1,1,F]
		tproj = grid_unsq * freqs  # [B,2,H,W,F]

		# sin/cos and pack -> [B, 4*num_frequencies, H, W]
		sin_feat = torch.sin(tproj)  # [B,2,H,W,F]
		cos_feat = torch.cos(tproj)
		fourier = torch.cat([sin_feat, cos_feat], dim=-1)  # [B,2,H,W,2F]

		B, C2, H_, W_, F2 = fourier.shape  # C2==2, F2==2*num_frequencies
		fourier_ch = fourier.permute(0, 1, 4, 2, 3).contiguous().view(B, C2 * F2, H_, W_)

		out = self.proj(fourier_ch)  # [B, pos_dim, H, W]
		return out
