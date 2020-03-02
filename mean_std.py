import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils import data

from lib.utils import *
from lib.dataset.seg_dataset import SegDataset
from tqdm import tqdm

global_path = '/home/kuowei/Downloads/img_seg'

#dataset_path = global_path + '/dataset/np_data/whole/data_list'
dataset_path = global_path + '/dataset/np_data/partition/partition_data_list'

train_dataset = SegDataset(dataset_path, split='train', is_norm=False)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=16, shuffle=False, drop_last=False)

sq_data_sum = 0
data_sum = 0
total_num = 0

for batch_count, (input_data, gt_data, input_name) in tqdm(enumerate(train_loader)):
	
	data = input_data.squeeze(0).numpy()

	_, x, y, z = data.shape
	
	sq_data_sum += np.sum(np.square(data))
	data_sum += (np.sum(data) / total_num)
	total_num += x*y*z

train_mean = data_sum / total_num
train_std = np.sqrt((sq_data_sum - total_num * np.square(train_mean)) / total_num)

print(train_mean)
print(train_std)
np.save('train_input_mean.npy', train_mean)
np.save('train_input_std.npy', train_std)
