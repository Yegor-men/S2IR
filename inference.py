from modules.render_image import render_image
import torch
from typing import Callable, Optional, Tuple

_eps = 1e-8


def ddim_step(
		x_t: torch.Tensor,
		eps_pred: torch.Tensor,
		a_t: torch.Tensor,
		a_s: torch.Tensor,
		eta: float = 0.0
) -> torch.Tensor:
	"""
	Single DDIM / ancestral-DDIM step from t -> s.

	Args:
		x_t: [B,C,H,W] current latent at timestep t
		eps_pred: [B,C,H,W] model-predicted noise at t (Gaussian)
		a_t: [B] or [B,1,1,1] alpha_bar(t)
		a_s: [B] or [B,1,1,1] alpha_bar(s)
		eta: 0.0 deterministic DDIM, >0 stochastic (ancestral)

	Returns:
		x_s: [B,C,H,W] next latent at timestep s
	"""
	device = x_t.device
	if a_t.dim() == 1:
		a_t = a_t.view(-1, 1, 1, 1).to(device)
	else:
		a_t = a_t.to(device)
	if a_s.dim() == 1:
		a_s = a_s.view(-1, 1, 1, 1).to(device)
	else:
		a_s = a_s.to(device)

	# 1) Predict x0 using properly scaled noise
	eps_scaled = eps_pred * torch.sqrt(1.0 - a_t)
	x0_hat = (x_t - eps_scaled) / (torch.sqrt(a_t) + _eps)

	# 2) Direction term
	sqrt_a_t = torch.sqrt(a_t)
	sqrt_1_a_t = torch.sqrt(1.0 - a_t)
	eps_dir = (x_t - sqrt_a_t * x0_hat) / (sqrt_1_a_t + _eps)

	# 3) Ancestral variance
	frac = ((1.0 - a_s) / (1.0 - a_t + _eps)).clamp(min=0.0)
	one_minus = (1.0 - (a_t / (a_s + _eps))).clamp(min=0.0)
	sigma = eta * torch.sqrt((frac * one_minus).clamp(min=0.0))

	# 4) Compose next latent
	sqrt_a_s = torch.sqrt(a_s)
	coef = torch.sqrt((1.0 - a_s - sigma ** 2).clamp(min=0.0))
	z = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

	x_s = sqrt_a_s * x0_hat + coef * eps_dir + sigma * z
	return x_s


def sample_ddim(
		model: torch.nn.Module,
		shape: Tuple[int, int, int, int],
		num_steps: int,
		alpha_bar_fn: Callable[[torch.Tensor], torch.Tensor],
		device: Optional[torch.device] = None,
		guidance: Optional[torch.Tensor] = None,
		guidance_uncond: Optional[torch.Tensor] = None,
		pos_cond: Optional[torch.Tensor] = None,
		cfg_scale: float = 1.0,
		eta: float = 0.0,
		render_intermediate: bool = True
) -> torch.Tensor:
	"""
	DDIM sampling loop with properly scaled noise for intermediate and final images.
	"""
	device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
	B, C, H, W = shape

	x = torch.randn(shape, device=device)

	if guidance is not None:
		guidance = guidance.to(device)
	if guidance_uncond is None and guidance is not None:
		guidance_uncond = torch.zeros_like(guidance, device=device)
	elif guidance_uncond is not None:
		guidance_uncond = guidance_uncond.to(device)

	if pos_cond is not None:
		pos_cond = pos_cond.to(device)

	ts = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)

	model.eval()
	with torch.no_grad():
		for i in range(num_steps):
			t = ts[i].expand(B)
			s = ts[i + 1].expand(B)
			a_t = alpha_bar_fn(t).to(device)
			a_s = alpha_bar_fn(s).to(device)

			# Predict eps
			if cfg_scale != 1.0 and guidance is not None:
				eps_uncond = model(x, guidance_uncond, pos_cond, a_t)
				eps_cond = model(x, guidance, pos_cond, a_t)
				eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
			else:
				eps_pred = model(x, guidance, pos_cond, a_t)

			# Compute x0_hat for rendering (scale model output)
			eps_scaled = eps_pred * torch.sqrt(1.0 - a_t.view(-1, 1, 1, 1))
			x0_hat = (x - eps_scaled) / (torch.sqrt(a_t.view(-1, 1, 1, 1)) + _eps)

			if render_intermediate:
				samples_display = (x0_hat + 1.0) / 2.0
				render_image(samples_display.clamp(0.0, 1.0))

			# Step to next latent
			x = ddim_step(x, eps_pred, a_t, a_s, eta=eta)

		# Final x0_hat
		eps_pred_final = model(x, guidance, pos_cond, a_s)
		if cfg_scale != 1.0 and guidance is not None:
			eps_uncond_final = model(x, guidance_uncond, pos_cond, a_s)
			eps_cond_final = model(x, guidance, pos_cond, a_s)
			eps_pred_final = eps_uncond_final + cfg_scale * (eps_cond_final - eps_uncond_final)

		# Scale final noise
		eps_final_scaled = eps_pred_final * torch.sqrt(1.0 - a_s.view(-1, 1, 1, 1))
		x0_hat_final = (x - eps_final_scaled) / (torch.sqrt(a_s.view(-1, 1, 1, 1)) + _eps)

		# Render final image
		samples_display_final = (x0_hat_final + 1.0) / 2.0
		render_image(samples_display_final.clamp(0.0, 1.0))

	return x0_hat_final


# ======================================================================================================================
from modules.alpha_bar import alpha_bar_cosine
from modules.relative_positional_conditioning import relative_positional_conditioning
from modules.global_embed import global_embed

alpha_bar_fn = alpha_bar_cosine

B, C, H, W = 9, 1, 28, 28

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.s2ir import SIIR

model = SIIR(
	c_channels=1,
	d_channels=64,
	num_heads=4,
	num_blocks=6,
	t_dim=128,
	text_embed_dim=10,
	pos_embed_dim=2,
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/s2ir_20250909_205256.pt", "cuda")
model.to(device)
model.eval()

label = torch.zeros(10)
label[0] = 1.0
labels = label.unsqueeze(0).expand(B, -1).to(device)

text_cond = global_embed(labels, H, W).to(device)
text_cond_uncond = torch.zeros_like(text_cond).to(device)
pos_cond = relative_positional_conditioning(torch.zeros(B, C, H, W)).to(device)

samples = sample_ddim(
	model=model,
	shape=(B, C, H, W),
	num_steps=50,
	alpha_bar_fn=alpha_bar_cosine,
	device=device,
	guidance=text_cond,
	guidance_uncond=text_cond_uncond,
	pos_cond=pos_cond,
	cfg_scale=1.5,
	eta=0.5,
)

# Postprocess for display (if you trained on [-1,1]):
samples_display = (samples + 1.0) / 2.0
samples_display = samples_display.clamp(0.0, 1.0)

render_image(samples_display)
render_image(torch.randn(B, C, H, W))
