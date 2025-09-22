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

from modules.s2ir import SIIR
from modules.count_params import count_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

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

model = load_checkpoint_into(model, "models/foo_04534.pt", "cuda")
model.to(device)
model.eval()

# ======================================================================================================================
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from tqdm import tqdm
import numpy as np

max_num = 500

losses = []

with torch.no_grad():
	t_range = torch.linspace(0, 1.0, steps=max_num)
	t_scrape_mse_losses = []
	t_scrape_snr_losses = []

	for _, t in tqdm(enumerate(t_range), total=max_num):
		image, label = next(iter(train_data))
		image, label = (image * 2.0 - 1.0), label
		b, c, h, w = image.shape

		time_vector = torch.ones(b) * t
		alpha_bar = alpha_bar_cosine(time_vector)
		noisy_image, expected_output = corrupt_image(image, alpha_bar)
		cfg_mask = torch.rand(b).round().unsqueeze(-1)
		text_cond = label * cfg_mask

		noisy_image = noisy_image.to(device)
		expected_output = expected_output.to(device)
		text_cond = text_cond.to(device)
		alpha_bar = alpha_bar.to(device)

		snr = alpha_bar / (1 - alpha_bar)
		weight = (1 / (snr + 1e-6)).clamp_(max=100.0)
		print(f"{t} = {torch.max(weight)}")

		predicted_noise = model(noisy_image, text_cond, alpha_bar)
		mse_loss = nn.functional.mse_loss(predicted_noise, expected_output)
		snr_loss = (weight * ((predicted_noise - expected_output) ** 2).mean(dim=[1, 2, 3])).mean()
		t_scrape_mse_losses.append(mse_loss.item())
		t_scrape_snr_losses.append(snr_loss.item())

# x = np.linspace(0, 1, len(t_scrape_losses))
# plt.plot(x, t_scrape_losses)
# percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# indices = [int(p / 100 * (len(t_scrape_losses) - 1)) for p in percentiles]
# percentile_x = [x[i] for i in indices]
# percentile_y = [t_scrape_losses[i] for i in indices]
# for px, py, p in zip(percentile_x, percentile_y, percentiles):
# 	plt.scatter(px, py, color='red')
# 	plt.text(px, py, f'{py}', fontsize=9, ha='center', va='bottom')
# plt.title('T scrape Losses')
#
# for i, v in enumerate(percentile_y):
# 	print(f"t = {percentiles[i] / 100}, loss = {v:.6f}")

plt.plot(t_scrape_mse_losses, label="MSE")
plt.plot(t_scrape_snr_losses, label="SNR")
# plt.plot(losses)
plt.legend()
plt.show()
