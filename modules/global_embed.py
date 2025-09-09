import torch


def global_embed(x: torch.Tensor, h: int, w: int):
	return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
