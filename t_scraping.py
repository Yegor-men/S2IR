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
	d_channels=128,
	num_heads=4,
	num_blocks=6,
	t_dim=128,
	text_embed_dim=10,
	pos_embed_dim=2,
	dropout_p_ffw=0.2,
	dropout_p_axial=0.1,
	dropout_p_cross=0.1,
).to(device)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/ema_05688_gauss_noise.pt", "cuda")
model.to(device)
model.eval()

# ======================================================================================================================
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from modules.global_embed import global_embed
from modules.render_image import render_image
from modules.relative_positional_conditioning import relative_positional_conditioning
from tqdm import tqdm

max_num = 1000

losses = []

with torch.no_grad():
	for i in tqdm(range(max_num), total=max_num, leave=True, desc=f"Iterating over t"):
		image, label = next(iter(train_data))
		image, label = (image * 2.0 - 1.0).to(device), label.to(device)
		b, c, h, w = image.shape

		t = i / max_num
		time_vector = torch.full((b,), t).to(device)
		alpha_bar = alpha_bar_cosine(time_vector)
		noisy_image, expected_output = corrupt_image(image, alpha_bar)
		cfg_mask = torch.rand(b).round().view(b, 1, 1, 1).to(device)
		text_cond = global_embed(label, h, w) * cfg_mask
		pos_cond = relative_positional_conditioning(image).to(device)

		predicted = model(noisy_image, text_cond, pos_cond, alpha_bar)
		loss = nn.functional.mse_loss(predicted, expected_output)
		losses.append(loss.item())

		if i in {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999}:
			print(f"loss.item(): {loss.item()}")
			fixed_noisy = torch.clamp(((noisy_image + 1) / 2), min=0.0, max=1.0)
			fixed_predicted = torch.clamp(((predicted + 1) / 2), min=0.0, max=1.0)
			fixed_expected = torch.clamp(((expected_output + 1) / 2), min=0.0, max=1.0)

			render_image(fixed_noisy, title=f"T{t} - Noisy Image")
			render_image(fixed_predicted, title=f"T{t} - Predicted")
			render_image(fixed_expected, title=f"T{t} - Expected")

plt.plot(losses)
plt.show()
