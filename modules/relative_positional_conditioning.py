import torch


def relative_positional_conditioning(x: torch.Tensor):
	B, C, H, W = x.shape

	if H >= W:
		y_coords = torch.linspace(-1.0, 1.0, steps=H)
		x_extent = (W / H)
		x_coords = torch.linspace(-x_extent, x_extent, steps=W)
	else:
		x_coords = torch.linspace(-1.0, 1.0, steps=W)
		y_extent = (H / W)
		y_coords = torch.linspace(-y_extent, y_extent, steps=H)

	yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # each [H, W]

	pos_hw = torch.stack([yy, xx], dim=0)

	pos = pos_hw.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

	return pos
