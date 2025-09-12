import torch
from torch import nn
import math


class RelPosEmbed(nn.Module):
	def __init__(
			self,
			pos_dim: int = 64,
			num_frequencies: int = 32,
			max_period: float = 10000.0
	):
		super().__init__()

		self.pos_dim = pos_dim
		self.num_frequencies = num_frequencies
		self.max_period = float(max_period)

		frequencies = torch.exp(torch.linspace(0.0, math.log(self.max_period), self.num_frequencies))
		self.register_buffer("frequencies", frequencies * (2.0 * math.pi), persistent=False)

		in_ch = 4 * 2 * self.num_frequencies

		self.proj = nn.Sequential(
			nn.Conv2d(in_ch, max(pos_dim, in_ch), kernel_size=1),
			nn.SiLU(),
			nn.Conv2d(max(pos_dim, in_ch), pos_dim, kernel_size=1)
		)

	def _make_base_4(self, h: int, w: int):
		if w >= h:
			x_min, x_max = -1.0, 1.0
			y_extent = h / w
			y_min, y_max = -y_extent, y_extent
		else:
			y_min, y_max = -1.0, 1.0
			x_extent = w / h
			x_min, x_max = -x_extent, x_extent

		x_coords = torch.linspace(x_min, x_max, steps=w, dtype=torch.float32)
		y_coords = torch.linspace(y_min, y_max, steps=h, dtype=torch.float32)

		yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

		left = (xx - x_min) / (x_max - x_min)
		right = 1.0 - left

		top = (yy - y_min) / (y_max - y_min)
		bottom = 1.0 - top

		base = torch.stack([left, right, top, bottom], dim=0)
		return base

	def forward(self, b: int, h: int, w: int) -> torch.Tensor:
		device = self.frequencies.device

		base4 = self._make_base_4(h, w).to(device)

		base4 = base4.unsqueeze(0).expand(b, -1, -1, -1)

		B4 = base4.unsqueeze(-1)
		freqs_sh = self.frequencies.view(1, 1, 1, 1, -1)
		tproj = B4 * freqs_sh

		sin_feat = torch.sin(tproj)
		cos_feat = torch.cos(tproj)

		fourier = torch.cat([sin_feat, cos_feat], dim=-1)

		B_, C4, H_, W_, F = fourier.shape
		fourier_ch = fourier.permute(0, 1, 4, 2, 3).contiguous().view(B_, C4 * F, H_, W_)

		out = self.proj(fourier_ch)
		return out
