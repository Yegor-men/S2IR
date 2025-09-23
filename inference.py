# ======================================================================================================================
import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image

B, C, H, W = 100, 1, 28, 28
device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.s2ir import SIIR

model = SIIR(
	c_channels=1,
	d_channels=32,
	time_freq=8,
	pos_freq=5,
	num_blocks=6,
	num_heads=2,
	text_cond_dim=10,
	text_token_length=1,
	cross_dropout=0.05,
	axial_dropout=0.05,
	ffn_dropout=0.2,
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/foo_sqrt_20_039.pt", "cuda")
model.to(device)
model.eval()

positive_text_conditioning = torch.zeros(100, 10)
for i in range(10):
	positive_text_conditioning[i * 10:(i + 1) * 10, i] = 1.0

# positive_text_conditioning = torch.zeros(B, 10)
# positive_text_conditioning[:, 6] = 1.0

initial_noise = torch.randn(B, C, H, W)
zero_text_conditioning = torch.zeros_like(positive_text_conditioning).to(device)
render_image((initial_noise + 1) / 2)
# ======================================================================================================================
import torch
import math
from typing import Optional


@torch.no_grad()
def run_ddim_visualization(
		model: torch.nn.Module,
		initial_noise: torch.Tensor,
		positive_text_conditioning: torch.Tensor,
		zero_text_conditioning: torch.Tensor,
		alpha_bar_fn,
		render_image_fn=None,
		num_steps: int = 50,
		cfg_scale: float = 1.0,  # conservative default for debugging
		eta: float = 0.0,
		render_every: int = 1,
		start_t: float | None = None,  # if None choose safe default below
		device: Optional[torch.device] = None,
):
	device = device or initial_noise.device
	model = model.to(device)
	model.eval()

	x = initial_noise.to(device)
	B, C, H, W = x.shape
	cond = positive_text_conditioning.to(device)
	uncond = zero_text_conditioning.to(device)

	# safe default for start_t:
	if start_t is None:
		# default: prefer at least 0.9; but also allow num_steps/(num_steps+1) if > 0.9
		start_guess = float(num_steps) / float(num_steps + 1)
		start_t = max(0.9, start_guess)

	# create timesteps descending from start_t -> 0 with num_steps+1 points
	ts = torch.linspace(start_t, 0.0, steps=(num_steps + 1), device=device)

	eps_small = 1e-6

	# initial render
	if render_image_fn is not None:
		render_image_fn(torch.clamp((x + 1.0) / 2.0, 0.0, 1.0))

	for i in range(num_steps):
		t_val = float(ts[i].item())
		s_val = float(ts[i + 1].item())

		t_batch = torch.full((B,), fill_value=t_val, device=device, dtype=torch.float32)
		s_batch = torch.full((B,), fill_value=s_val, device=device, dtype=torch.float32)

		# evaluate alpha_bar (and defend against exact zero)
		a_t = alpha_bar_fn(t_batch).to(device).clamp(min=eps_small)
		a_s = alpha_bar_fn(s_batch).to(device).clamp(min=eps_small)

		# model outputs (classifier-free guidance: uncond + cond)
		eps_uncond = model(x, uncond, a_t)  # shape [B,C,H,W]
		eps_cond = model(x, cond, a_t)
		eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

		# compute x0_hat stably
		a_t_view = a_t.view(B, 1, 1, 1)
		sqrt_a_t = torch.sqrt(a_t_view)
		sqrt_1_a_t = torch.sqrt((1.0 - a_t_view).clamp(min=0.0))
		x0_hat = (x - sqrt_1_a_t * eps_hat) / (sqrt_a_t + eps_small)

		# render reconstruction
		if (i % render_every == 0) and (render_image_fn is not None):
			render_image_fn(torch.clamp((x0_hat + 1.0) / 2.0, 0.0, 1.0))

		# debug prints for first few steps
		if i < 4:
			print(f"step {i}: t={t_val:.6f} -> s={s_val:.6f}")
			print(f"  a_t mean {a_t.mean().item():.3e}  min {a_t.min().item():.3e}")
			print(f"  eps_hat mean {eps_hat.mean().item():.3f}  std {eps_hat.std().item():.3f}")
			print(f"  x min/max {x.min().item():.3f}/{x.max().item():.3f}"
				  f"  x0_hat min/max {x0_hat.min().item():.3f}/{x0_hat.max().item():.3f}")

		# direction and ancestral sigma
		eps_dir = (x - sqrt_a_t * x0_hat) / (sqrt_1_a_t + eps_small)

		a_s_view = a_s.view(B, 1, 1, 1)
		ratio = ((1.0 - a_s) / (1.0 - a_t + eps_small)).clamp(min=0.0)
		coef = (1.0 - (a_t / (a_s + eps_small))).clamp(min=0.0)
		sigma = eta * torch.sqrt((ratio * coef).clamp(min=0.0)).view(B, 1, 1, 1)

		base_noise_sq = (1.0 - a_s_view - sigma * sigma).clamp(min=0.0)
		base_noise_scale = torch.sqrt(base_noise_sq)

		z = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)

		# update x
		x = torch.sqrt(a_s_view) * x0_hat + base_noise_scale * eps_dir + sigma * z

	# final reconstruction
	final_t = torch.zeros((B,), device=device, dtype=torch.float32)
	final_a = alpha_bar_fn(final_t).to(device).view(B, 1, 1, 1).clamp(min=eps_small)
	final_x0_hat = (x - torch.sqrt((1.0 - final_a).clamp(min=0.0)) * eps_hat) / (torch.sqrt(final_a) + eps_small)

	if render_image_fn is not None:
		render_image_fn(torch.clamp((final_x0_hat + 1.0) / 2.0, 0.0, 1.0))

	return final_x0_hat, x


final_x0_hat, final_x = run_ddim_visualization(
	model=model,
	initial_noise=initial_noise,
	positive_text_conditioning=positive_text_conditioning,
	zero_text_conditioning=zero_text_conditioning,
	alpha_bar_fn=alpha_bar_cosine,
	render_image_fn=render_image,
	num_steps=20,
	cfg_scale=1.5,  # safe
	eta=1.0,
	render_every=1,
	start_t=0.999,  # explicit safe start
	device=torch.device("cuda")
)
