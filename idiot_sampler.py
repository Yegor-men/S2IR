# ======================================================================================================================
import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.relative_positional_conditioning import relative_positional_conditioning
from modules.global_embed import global_embed
from modules.render_image import render_image

B, C, H, W = 9, 1, 28, 28

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.s2ir import SIIR

model = SIIR(
	c_channels=1,
	d_channels=64,
	num_heads=4,
	num_blocks=4,
	t_dim=16,
	text_embed_dim=10,
	pos_embed_dim=2,
	dropout_p_ffw=0.33,
	dropout_p_axial=0.15,
	dropout_p_cross=0.15,
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/noise_sub_img.pt", "cuda")
model.to(device)
model.eval()

label = torch.zeros(10)
label[0] = 1.0
labels = label.unsqueeze(0).expand(B, -1).to(device)

noisy_image = torch.randn(B, C, H, W).to(device)
positive_text_conditioning = global_embed(labels, H, W).to(device)
# zero_text_conditioning = torch.zeros_like(positive_text_conditioning).to(device)
position_conditioning = relative_positional_conditioning(noisy_image).to(device)
render_image(((noisy_image + 1) / 2).clamp(min=0.0, max=1))
# ======================================================================================================================

num_diffusion_steps = 10
diffusion_per_step = torch.arange(1, num_diffusion_steps + 1, 1)
diffusion_per_step = 1 / diffusion_per_step
diffusion_per_step = diffusion_per_step / torch.sum(diffusion_per_step)
print(diffusion_per_step.numel())

mix_percent = 1
alpha_bar = torch.zeros(B).to(device)

with torch.no_grad():
	for i in range(num_diffusion_steps):
		diffusion_amount = diffusion_per_step[i].item()

		positive = model(
			noisy_image,
			positive_text_conditioning,
			position_conditioning,
			alpha_bar
		)

		# negative = model(
		# 	noisy_image,
		# 	zero_text_conditioning,
		# 	position_conditioning,
		# 	alpha_bar
		# )

		# final_prediction = positive * mix_percent + negative * (1 - mix_percent)

		noisy_image -= diffusion_amount * positive

		alpha_bar += diffusion_amount

		render_image(((noisy_image + 1) / 2).clamp(min=0.0, max=1))
