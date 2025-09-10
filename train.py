import matplotlib.pyplot as plt
import torch
from torch import nn
# ======================================================================================================================
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 100
NUM_CLASSES = 10


def one_hot_encode(label, num_classes=NUM_CLASSES):
	return torch.nn.functional.one_hot(torch.tensor(label), num_classes=num_classes).float()


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

train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

from modules.s2ir import SIIR
from modules.count_params import count_parameters

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
).to(device)

count_parameters(model)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

# ======================================================================================================================
from tqdm import tqdm
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from modules.relative_positional_conditioning import relative_positional_conditioning
from modules.render_image import render_image
from modules.global_embed import global_embed

num_epochs = 10
train_losses = []
test_losses = []

for E in range(num_epochs):
	model.train()
	train_loss = 0
	for i, (image, label) in tqdm(enumerate(train_data), total=len(train_data), leave=True, desc=f"E: {E}"):
		optimizer.zero_grad()
		image, labels = (image * 2.0 - 1.0).to(device), label.to(device)
		b, c, h, w = image.shape
		with torch.no_grad():
			times = torch.rand(b).to(device)
			alpha_bar = alpha_bar_cosine(times)
			noisy_image, expected_output = corrupt_image(image, alpha_bar)
			cfg_mask = torch.rand(b).round().view(b, 1, 1, 1).to(device)
			text_cond = global_embed(labels, h, w) * cfg_mask
			pos_cond = relative_positional_conditioning(image).to(device)

		predicted_noise = model(noisy_image, text_cond, pos_cond, alpha_bar)
		loss = nn.functional.mse_loss(predicted_noise, expected_output)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()

	train_loss /= len(train_data)
	train_losses.append(train_loss)

	model.eval()
	test_loss = 0
	with torch.no_grad():
		for i, (image, label) in tqdm(enumerate(test_data), total=len(test_data), leave=True, desc=f"T: {E}"):
			image, labels = (image * 2.0 - 1.0).to(device), label.to(device)
			b, c, h, w = image.shape

			times = torch.rand(b).to(device)
			alpha_bar = alpha_bar_cosine(times)
			noisy_image, expected_output = corrupt_image(image, alpha_bar)
			cfg_mask = torch.rand(b).round().view(b, 1, 1, 1).to(device)
			text_cond = global_embed(labels, h, w) * cfg_mask
			pos_cond = relative_positional_conditioning(image).to(device)

			predicted_noise = model(noisy_image, text_cond, pos_cond, alpha_bar)
			loss = nn.functional.mse_loss(predicted_noise, expected_output)
			test_loss += loss.item()
			# print(loss.item())

			if i == 0:
				fixed_noisy = torch.clamp(((noisy_image + 1) / 2), min=0.0, max=1.0)
				fixed_predicted = torch.clamp(((predicted_noise + 1) / 2), min=0.0, max=1.0)
				fixed_expected = torch.clamp(((expected_output + 1) / 2), min=0.0, max=1.0)

				render_image(fixed_noisy, title=f"E{E} - Noisy Image")
				render_image(fixed_predicted, title=f"E{E} - Predicted")
				render_image(fixed_expected, title=f"E{E} - Expected")
				render_image((predicted_noise - expected_output) ** 2, title=f"E{E} - Squared Error")

		test_loss /= len(test_data)
		test_losses.append(test_loss)

	plt.plot(train_losses, label="train")
	plt.plot(test_losses, label="test")
	plt.legend()
	plt.show()
	print(f"\nEpoch {E} - TRAIN: {train_loss:.5f}, TEST: {test_loss:.5f}")

# ======================================================================================================================
from save_load_model import save_checkpoint

model_path = save_checkpoint(model)
