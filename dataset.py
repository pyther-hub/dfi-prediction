from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

import os


class CustomImageDataset(Dataset):
	def __init__(self, img_dir, df, transform=None):
		self.img_dir = img_dir
		self.img_name = df['file_name'].values
		self.img_labels = df['label'].values.astype(float)
		self.transform = transform

	def __len__(self):
		return len(self.img_name)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_name[idx])
		image = Image.open(img_path).convert("RGB")
		label = self.img_labels[idx]
		if self.transform:
			image = self.transform(image)
		return image, label