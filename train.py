import time

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from save_load_model import save_checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ======================================================================================================================
def one_hot_encode(label):
	return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotMNIST(torch.utils.data.Dataset):
	def __init__(self, train=True):
		self.dataset = datasets.MNIST(
			root='data',
			train=train,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
			])
		)

	def __getitem__(self, index):
		image, label = self.dataset[index]
		one_hot_label = one_hot_encode(label)
		return image, one_hot_label

	def __len__(self):
		return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)

# ======================================================================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

from modules.s2ir import SIIR
from modules.count_params import count_parameters

model = SIIR(
	c_channels=1,
	d_channels=32,
	time_freq=8,
	pos_freq=5,
	num_blocks=8,
	num_heads=2,
	text_cond_dim=10,
	text_token_length=1,
	cross_dropout=0.1,
	axial_dropout=0.1,
	ffn_dropout=0.2,
).to(device)

# from save_load_model import load_checkpoint_into
#
# model = load_checkpoint_into(model, "models/02021_baseline.pt", "cuda")
# model.to(device)
# model.eval()

import copy

ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
	param.requires_grad = False


@torch.no_grad()
def update_ema_model(model, ema_model, decay):
	for param, ema_param in zip(model.parameters(), ema_model.parameters()):
		ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


count_parameters(model)

# ======================================================================================================================
import math
from torch.optim.lr_scheduler import LambdaLR


def make_cosine_with_warmup(optimizer, warmup_steps, total_steps, lr_end):
	peak_lr = float(optimizer.defaults['lr'])

	lr_end = float(lr_end)
	min_mult = lr_end / peak_lr

	def lr_lambda(step):
		step = float(step)
		if step <= 0:
			return max(min_mult, 0.0)
		if step < warmup_steps:
			return (step / float(max(1.0, warmup_steps)))
		# after warmup: cosine decay from 1.0 -> min_mult
		progress = (step - warmup_steps) / float(max(1.0, total_steps - warmup_steps))
		progress = min(1.0, max(0.0, progress))
		cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
		# map cosine in [0,1] to multiplier in [min_mult, 1.0]
		return min_mult + (1.0 - min_mult) * cosine

	return LambdaLR(optimizer, lr_lambda, -1)


def bicubic_rescale(tensor, new_height, new_width):
	return torch.nn.functional.interpolate(
		tensor,
		size=(new_height, new_width),
		mode='bicubic',
		align_corners=False
	)


num_epochs = 15
batch_size = 50
ema_decay = 0.999

train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

peak_lr = 1e-3
final_lr = 1e-5
total_steps = num_epochs * len(train_dloader)
warmup_steps = 1200

optimizer = torch.optim.AdamW(params=model.parameters(), lr=peak_lr)
scheduler = make_cosine_with_warmup(optimizer, warmup_steps, total_steps, final_lr)

# ======================================================================================================================

from tqdm import tqdm
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from modules.render_image import render_image

train_losses = []
test_losses = []
percentile_losses = []

scales = [14, 27, 40, 51, 64]

for E in range(num_epochs):

	# TRAINING
	train_loss = 0
	model.train()
	for i, (orig_image, label) in tqdm(enumerate(train_dloader), total=len(train_dloader), leave=True, desc=f"E: {E}"):
		b, c, h, w = orig_image.shape
		if b != batch_size:
			continue
		batch_loss = 0
		for scale in scales:
			with torch.no_grad():
				image = orig_image * 2.0 - 1.0
				image = bicubic_rescale(image, scale, scale)
				# t = torch.distributions.Beta(2.0, 2.0).sample([b])
				t = torch.rand(b)
				alpha_bar = alpha_bar_cosine(t)
				noisy_image, expected_output = corrupt_image(image, alpha_bar)
				cfg_mask = torch.rand(b).round().unsqueeze(-1)
				text_cond = label * cfg_mask

				noisy_image = noisy_image.to(device)
				expected_output = expected_output.to(device)
				text_cond = text_cond.to(device)
				alpha_bar = alpha_bar.to(device)

			predicted_noise = model(noisy_image, text_cond, alpha_bar)
			loss = nn.functional.mse_loss(predicted_noise, expected_output)
			batch_loss += (loss / len(scales)).item()
			(loss / len(scales)).backward()

		train_loss += batch_loss
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()
		update_ema_model(model, ema_model, ema_decay)

	train_loss /= len(train_dloader)
	train_losses.append(train_loss)

	# TESTING
	test_loss = 0
	model.eval()
	with torch.no_grad():
		for i, (orig_image, label) in tqdm(enumerate(test_dloader), total=len(test_dloader), leave=True,
										   desc=f"E: {E}"):
			b, c, h, w = orig_image.shape
			if b != batch_size:
				continue
			batch_loss = 0
			for j, scale in enumerate(scales):
				image = orig_image * 2.0 - 1.0
				image = bicubic_rescale(image, scale, scale)
				# t = torch.distributions.Beta(2.0, 2.0).sample([b])
				t = torch.rand(b)
				alpha_bar = alpha_bar_cosine(t)
				noisy_image, expected_output = corrupt_image(image, alpha_bar)
				cfg_mask = torch.rand(b).round().unsqueeze(-1)
				text_cond = label * cfg_mask

				noisy_image = noisy_image.to(device)
				expected_output = expected_output.to(device)
				text_cond = text_cond.to(device)
				alpha_bar = alpha_bar.to(device)

				predicted_noise = ema_model(noisy_image, text_cond, alpha_bar)
				loss = nn.functional.mse_loss(predicted_noise, expected_output)
				batch_loss += (loss / len(scales)).item()

				if i == 0 and j == 3:
					fixed_noisy = torch.clamp(((noisy_image + 1) / 2), min=0.0, max=1.0)
					fixed_predicted = torch.clamp(((predicted_noise + 1) / 2), min=0.0, max=1.0)
					fixed_expected = torch.clamp(((expected_output + 1) / 2), min=0.0, max=1.0)

					render_image(fixed_noisy, title=f"E{E} - S{scale} - Noisy Image")
					render_image(fixed_predicted, title=f"E{E} - S{scale} - Predicted")
					render_image(fixed_expected, title=f"E{E} - S{scale} - Expected")
					render_image((predicted_noise - expected_output) ** 2, title=f"E{E} - S{scale} - Squared Error")

			test_loss += batch_loss

	test_loss /= len(test_dloader)
	test_losses.append(test_loss)
	print(f"Epoch {E} - TRAIN: {train_loss:.5f}, TEST: {test_loss:.5f}")
	time.sleep(0.5)

	plt.plot(train_losses, label="Train")
	plt.plot(test_losses, label="Test")
	plt.legend()
	plt.show()

	# T SCRAPE LOSSES
	with torch.no_grad():
		t_range = torch.linspace(0, 1, steps=200)
		t_scrape_losses = []

		for t in t_range:
			t_loss = 0
			for scale in scales:
				orig_image, label = next(iter(train_dloader))
				b, c, h, w = orig_image.shape
				image = orig_image * 2.0 - 1.0
				image = bicubic_rescale(image, scale, scale)

				alpha_bar = alpha_bar_cosine(torch.ones(b) * t)
				noisy_image, expected_output = corrupt_image(image, alpha_bar)
				cfg_mask = torch.rand(b).round().unsqueeze(-1)
				text_cond = label * cfg_mask

				noisy_image = noisy_image.to(device)
				expected_output = expected_output.to(device)
				text_cond = text_cond.to(device)
				alpha_bar = alpha_bar.to(device)

				predicted_noise = ema_model(noisy_image, text_cond, alpha_bar)
				loss = nn.functional.mse_loss(predicted_noise, expected_output)
				t_loss += (loss / len(scales)).item()

			t_scrape_losses.append(t_loss)

		x = np.linspace(0, 1, len(t_scrape_losses))
		plt.plot(x, t_scrape_losses)
		percentiles = [1, 25, 50, 75, 100]
		indices = [int(p / 100 * (len(t_scrape_losses) - 1)) for p in percentiles]
		percentile_x = [x[i] for i in indices]
		percentile_y = [t_scrape_losses[i] for i in indices]
		for px, py, p in zip(percentile_x, percentile_y, percentiles):
			plt.scatter(px, py, color='red')
			plt.text(px, py, f'{py}', fontsize=9, ha='center', va='bottom')
		plt.title('T scrape Losses')
		plt.show()

		percentile_losses.append(percentile_y)
		transposed = list(zip(*percentile_losses))
		for i, series in enumerate(transposed):
			plt.plot(series, label=f"t = {(percentiles[i] / 100):.2f}")
		plt.title("T scrape percentile losses over time")
		plt.legend()
		plt.show()

	# MODEL SAVING
	if (E + 1) % 1 == 0 or E == num_epochs:
		model_path = save_checkpoint(ema_model, prefix=f"E{E + 1}_{test_loss:.5f}_{percentile_y[2]}")
		time.sleep(0.5)
# ======================================================================================================================
