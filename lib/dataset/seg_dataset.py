import os
import sys

import torch
from torch.utils import data

import numpy as np


class SegDataset(data.Dataset):

	def __init__(self, root, split='train', is_transform=True, augmentations=None, is_norm=False):
		self.root = root
		self.split = split
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.is_norm = is_norm

		self.input_mean = None
		self.input_std = None

		if self.is_norm:

			input_mean_name = 'train_input_mean.npy'
			input_std_name = 'train_input_std.npy'

			self.input_mean = np.load(os.path.join(self.root, input_mean_name))
			self.input_std = np.load(os.path.join(self.root, input_std_name))


		#get data path
		self.input_list_file = os.path.join(self.root, split + '_input.txt')
		self.gt_list_file = os.path.join(self.root, split + '_gt.txt')

		self.input_data = []
		self.gt_data = []
		
		with open(self.input_list_file, 'r') as f:
			for line in f:
				self.input_data.append(line.rstrip('\n'))
		with open(self.gt_list_file, 'r') as f:
			for line in f:
				self.gt_data.append(line.rstrip('\n'))

	def __len__(self):
		return len(self.input_data)

	def __getitem__(self, index):

		input_path = self.input_data[index]
		input_name = input_path.split('/')[-1]

		gt_path = self.gt_data[index]

		input_tensor = np.load(input_path)
		gt_tensor = np.load(gt_path)

		if self.is_norm:	
			input_tensor = input_tensor - self.input_mean
			input_tensor = input_tensor / self.input_std

		input_tensor = input_tensor[np.newaxis, :, :, :]
		gt_tensor = gt_tensor[np.newaxis, :, :, :]

		return input_tensor, gt_tensor, input_name
