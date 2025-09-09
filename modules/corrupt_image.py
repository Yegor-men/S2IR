import torch


def corrupt_image(image_tensor, alpha_bar):
	B, C, H, W = image_tensor.shape
	alpha_bar = alpha_bar.view(B, 1, 1, 1)

	sqrt_ab = torch.sqrt(alpha_bar)
	sqrt_1_ab = torch.sqrt(1.0 - alpha_bar)

	eps = torch.randn_like(image_tensor)
	x_t = sqrt_ab * image_tensor + sqrt_1_ab * eps
	return x_t, eps
