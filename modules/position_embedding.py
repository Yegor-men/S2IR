# rel_pos_embed2d.py
import math
import torch
from torch import nn


class RelPosEmbed2D(nn.Module):
	def __init__(
			self,
			num_frequencies: int = 16,
			d_channels: int = 64,
			eps: float = 1e-6
	):
		super().__init__()
		self.pos_dim = int(d_channels)
		self.num_frequencies = int(num_frequencies)
		self.eps = float(eps)

		powers = torch.arange(self.num_frequencies, dtype=torch.float32)  # [0, 1, ...]
		freqs = math.pi * (2.0 ** powers)  # [pi, 2pi, 4pi, ...]
		self.register_buffer("frequencies", freqs, persistent=True)

		in_ch = 4 * self.num_frequencies

		# self.decay = nn.Parameter(torch.zeros(1))

		self.proj = nn.Sequential(
			nn.GroupNorm(1, in_ch),
			nn.Conv2d(in_channels=in_ch, out_channels=d_channels, kernel_size=1),
			nn.GroupNorm(1, d_channels)
		)

	def _make_grid(self, h: int, w: int):
		if w >= h:
			x_min, x_max = -0.5, 0.5
			y_extent = h / w
			y_min, y_max = -0.5 * y_extent, 0.5 * y_extent
		else:
			y_min, y_max = -0.5, 0.5
			x_extent = w / h
			x_min, x_max = -0.5 * x_extent, 0.5 * x_extent

		x_coords = torch.linspace(x_min + self.eps, x_max - self.eps, steps=w)
		y_coords = torch.linspace(y_min + self.eps, y_max - self.eps, steps=h)

		yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
		grid = torch.stack([xx, yy], dim=0)
		return grid

	def forward(self, h: int, w: int) -> torch.Tensor:
		grid = self._make_grid(h, w).to(self.frequencies)

		# min_pixels = (2 ** torch.arange(self.num_frequencies))
		# useful_mask = (min_pixels <= min(h, w)).to(self.frequencies)

		# base = torch.sigmoid(self.decay).clamp(min=1e-6)
		# powers = torch.arange(self.num_frequencies).to(self.frequencies)
		# decays = base ** powers

		# weights = decays * useful_mask

		grid_unsq = grid.unsqueeze(-1)  # [2, h, w, 1]
		freqs = self.frequencies.view(1, 1, 1, -1)  # [1, 1, 1, F]
		tproj = grid_unsq * freqs  # [2, h, w, F]

		# sin/cos and multiply amplitude weights -> sin_feat/cos_feat [2, h, w, F]
		# expand weights to broadcast: [1, 1, 1, F]
		# weights_view = weights.view(1, 1, 1, -1)
		# sin_feat = torch.sin(tproj) * weights_view
		# cos_feat = torch.cos(tproj) * weights_view
		sin_feat = torch.sin(tproj)
		cos_feat = torch.cos(tproj)

		# now rearrange into channel-first format expected by conv: [1, 4F, h, w]
		# sin_feat shape [2, h, w, F] -> permute -> [2, F, h, w] -> reshape [2F, h, w]
		sin_ch = sin_feat.permute(0, 3, 1, 2).contiguous().view(2 * self.num_frequencies, h, w)
		cos_ch = cos_feat.permute(0, 3, 1, 2).contiguous().view(2 * self.num_frequencies, h, w)
		fourier_ch = torch.cat([sin_ch, cos_ch], dim=0).unsqueeze(0)  # [1, 4F, h, w]

		out = self.proj(fourier_ch)  # [1, pos_dim, h, w]
		out = out.squeeze(0)  # -> [pos_dim, h, w]
		return out
