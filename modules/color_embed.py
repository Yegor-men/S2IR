import math
import torch
from torch import nn


class ColorEmbedding(nn.Module):
	def __init__(
			self,
			c_channels: int,
			d_channels: int,
			num_frequencies: int,
			v_scale: float = 6.0
	):
		super().__init__()
		self.c_channels = int(c_channels)
		self.d_channels = int(d_channels)
		self.num_frequencies = int(num_frequencies)
		self.v_scale = float(v_scale)

		# first frequency = pi/12 (half period across [-V,V]), then double each step
		powers = torch.arange(self.num_frequencies, dtype=torch.float32)
		base = math.pi / 12.0
		freqs = base * (2.0 ** powers)  # [pi/12, 2*pi/12, 4*pi/12, ...]
		self.register_buffer("frequencies", freqs, persistent=True)

		# now input channels = C * (2 * F) because we only use sin & cos (no raw scalar)
		in_ch = self.c_channels * (2 * self.num_frequencies)

		self.proj = nn.Sequential(
			nn.GroupNorm(1, in_ch),
			nn.Conv2d(in_channels=in_ch, out_channels=self.d_channels, kernel_size=1),
			# nn.GroupNorm(1, self.d_channels)
		)

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		"""
		image: [B, C, H, W] — expected to already be in the numeric range you work with (we clamp).
		returns: [B, d_channels, H, W]
		"""
		if image.dim() != 4:
			raise ValueError("image must be [B, C, H, W]")

		# clamp to stable value range
		x = image.clamp(min=-self.v_scale, max=self.v_scale)
		B, C, H, W = x.shape
		if C != self.c_channels:
			raise ValueError(f"expected input channels {self.c_channels}, got {C}")

		# frequencies shape -> [1, 1, 1, 1, F] on the same device/dtype as x
		freqs = self.frequencies.view(1, 1, 1, 1, -1).to(x)

		# tproj: [B, C, H, W, F] = x.unsqueeze(-1) * freqs
		tproj = x.unsqueeze(-1) * freqs  # broadcasting multiplication

		# sin / cos: [B, C, H, W, F]
		sin_feat = torch.sin(tproj)
		cos_feat = torch.cos(tproj)

		# move frequency dim into channel dimension: [B, C*F, H, W]
		# permute from [B, C, H, W, F] -> [B, C, F, H, W] then view
		sin_ch = sin_feat.permute(0, 1, 4, 2, 3).contiguous().view(B, C * self.num_frequencies, H, W)
		cos_ch = cos_feat.permute(0, 1, 4, 2, 3).contiguous().view(B, C * self.num_frequencies, H, W)

		# concatenate sin & cos -> [B, C*(2F), H, W]
		fourier_ch = torch.cat([sin_ch, cos_ch], dim=1)

		# project -> [B, d_channels, H, W]
		out = self.proj(fourier_ch)
		return out
