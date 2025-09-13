import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
# ======================================================================================================================
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
	d_channels=128,
	num_heads=4,
	num_blocks=6,
	time_freq=10,
	time_dim=64,
	pos_freq=6,
	pos_dim=64,
	text_cond_dim=10,
	cond_dropout=0.0,
	axial_dropout=0.0,
	ffn_dropout=0.0,
).to(device)

# from save_load_model import load_checkpoint_into
#
# model = load_checkpoint_into(model, "s2ir_14_09_25_04901.pt", "cuda")
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

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)

# ======================================================================================================================
num_epochs = 20
batch_size = 100
ema_decay = 0.999

train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

from tqdm import tqdm
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from modules.render_image import render_image
from modules.global_embed import global_embed

train_losses = []
test_losses = []
percentile_losses = []

for E in range(num_epochs):

	train_loss = 0
	model.train()

	for i, (image, label) in tqdm(enumerate(train_data), total=len(train_data), leave=True, desc=f"E: {E}"):
		b, c, h, w = image.shape

		with torch.no_grad():
			image, label = (image * 2.0 - 1.0).to(device), label
			alpha_bar = alpha_bar_cosine(torch.rand(b)).to(device)
			noisy_image, expected_output = corrupt_image(image, alpha_bar)
			cfg_mask = torch.rand(b).round().view(b, 1, 1, 1)
			text_cond = (global_embed(label, h, w) * cfg_mask).to(device)

		predicted = model(noisy_image, text_cond, alpha_bar)
		loss = nn.functional.mse_loss(predicted, expected_output)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		optimizer.zero_grad()
		cur_batch = E * len(train_data) + i
		decay = ema_decay if cur_batch > 1 / (1 - ema_decay) else cur_batch / (1 / (1 - ema_decay))
		update_ema_model(model, ema_model, ema_decay)

	train_loss /= len(train_data)
	train_losses.append(train_loss)

	test_loss = 0
	model.eval()

	with torch.no_grad():
		for i, (image, label) in tqdm(enumerate(test_data), total=len(test_data), leave=True, desc=f"T: {E}"):
			b, c, h, w = image.shape

			image, label = (image * 2.0 - 1.0).to(device), label
			alpha_bar = alpha_bar_cosine(torch.rand(b)).to(device)
			noisy_image, expected_output = corrupt_image(image, alpha_bar)
			cfg_mask = torch.rand(b).round().view(b, 1, 1, 1)
			text_cond = (global_embed(label, h, w) * cfg_mask).to(device)

			predicted = ema_model(noisy_image, text_cond, alpha_bar)
			loss = nn.functional.mse_loss(predicted, expected_output)
			test_loss += loss.item()

			if i == 0:
				fixed_noisy = torch.clamp(((noisy_image + 1) / 2), min=0.0, max=1.0)
				fixed_predicted = torch.clamp(((predicted + 1) / 2), min=0.0, max=1.0)
				fixed_expected = torch.clamp(((expected_output + 1) / 2), min=0.0, max=1.0)

				render_image(fixed_noisy, title=f"E{E} - Noisy Image")
				render_image(fixed_predicted, title=f"E{E} - Predicted")
				render_image(fixed_expected, title=f"E{E} - Expected")
				render_image((predicted - expected_output) ** 2, title=f"E{E} - Squared Error")

	test_loss /= len(test_data)
	test_losses.append(test_loss)

	plt.plot(train_losses, label="train")
	plt.plot(test_losses, label="test")
	plt.legend()
	plt.show()

	print(f"\nEpoch {E} - TRAIN: {train_loss:.5f}, TEST: {test_loss:.5f}")

	with torch.no_grad():
		t_range = torch.linspace(0, 1, steps=200)
		t_scrape_losses = []

		for t in t_range:
			image, label = next(iter(train_data))
			b, c, h, w = image.shape
			image, label = (image * 2.0 - 1.0).to(device), label

			alpha_bar = alpha_bar_cosine(torch.ones(b) * t).to(device)
			noisy_image, expected_output = corrupt_image(image, alpha_bar)
			cfg_mask = torch.rand(b).round().view(b, 1, 1, 1)
			text_cond = (global_embed(label, h, w) * cfg_mask).to(device)

			predicted = ema_model(noisy_image, text_cond, alpha_bar)
			loss = nn.functional.mse_loss(predicted, expected_output)

			t_scrape_losses.append(loss.item())

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

# ======================================================================================================================
from save_load_model import save_checkpoint

model_path = save_checkpoint(ema_model)
