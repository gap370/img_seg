import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from lib.dataset.seg_dataset import SegDataset

from lib.model.unet.unet import UNET
from lib.model.vdsrr.vdsrr import VDSRR

def prepareDataset(args, root_dir, data_aug=None, normalize=False):
	
	if args.dataset.lower() == 'seg':

		dataset_path = root_dir + 'img_seg/seg_dataset/np_data/partition/partition_data_list/'
		
		train_dataset = SegDataset(dataset_path, split='train', is_norm=normalize)
		val_dataset = SegDataset(dataset_path, split='validate', is_norm=normalize)
	
	else:
		raise ValueError('unknown dataset: ' + dataset)

	train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)
	val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=16, shuffle=False)

	print('Got {} training examples'.format(len(train_loader.dataset)))
	print('Got {} validation examples'.format(len(val_loader.dataset)))
	
	return train_loader, val_loader

def loadData(args, root_dir, prediction_data, normalize=False):

	if args.dataset == 'seg':
		
		prediction_set, subject_num, patch_num, case = prediction_data

		if case == 'whole':
			dataset_path = root_dir + 'img_seg/dataset/np_data/whole/data_list/'
		else:
			dataset_path = root_dir + 'img_seg/dataset/np_data/partition/partition_data_list/'

		if prediction_set == 'train':
			ext_data = SegDataset(dataset_path, split='train', is_norm=normalize)
		elif prediction_set == 'val':
			ext_data = SegDataset(dataset_path, split='validate', is_norm=normalize)
		elif prediction_set == 'test':
			ext_data = SegDataset(dataset_path, split='test', is_norm=normalize)
		#elif prediction_set == 'ext':
		#	ext_data = ext_handle(args, root_dir, prediction_data, normalize, dataset_path)
		#else:
		#	raise ValueError('Unknown extra data category: ' + prediction_set)

	else:
		raise ValueError('unknown dataset: ' + args.dataset)
		
	data_loader = data.DataLoader(ext_data, batch_size=1, num_workers=16, shuffle=False)
	
	print('Got {} testing examples'.format(len(data_loader.dataset)))

	return data_loader

def ext_handle(args, root_dir, prediction_data, normalize, dataset_path):
	pass
	#prediction_set, subject_num, ori_num, patch_num, case = prediction_data

	#input_name = 'ext_phase.txt'
	#gt_name = 'ext_gt.txt'

	#temp_sub = subject_num + '/' + ori_num

	#if case == 'whole':
	#	with open(dataset_path + input_name, 'w') as f:
	#		f.write(root_dir + 'qsm_dataset/qsm_B_z/mix_data/whole/phase_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_LBVSMV_rot.npy\n')
	#	with open(dataset_path + gt_name, 'w') as f:
	#		f.write(root_dir + 'qsm_dataset/qsm_B_z/mix_data/whole/cosmos_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_cosmos_rot.npy\n')
	
	#elif case == 'patch':
	#	with open(dataset_path + input_name, 'w') as f:
	#		f.write(root_dir + 'qsm_dataset/qsm_B_z/mix_data/partition/phase_pdata' + temp_sub + '/' + subject_num + '_' + ori_num + '_LBVSMV_rot_p' + patch_num + '.npy\n')
	#	with open(dataset_path + gt_name, 'w') as f:
	#		f.write(root_dir + 'qsm_dataset/qsm_B_z/mix_data/partition/cosmos_pdata/' + temp_sub + '/' + subject_num + '_' + ori_num + '_cosmos_rot_p' + patch_num + '.npy\n')
			
	#else:
	#	raise ValueError('unknown case: ' + case)

	ext_data = QsmDataset(dataset_path, split='ext', tesla=args.tesla, is_norm=normalize)

	return ext_data

def chooseModel(args):
	
	model = None
	if args.model_arch.lower() == 'vdsrr':
		model = VDSRR()
	elif args.model_arch.lower() == 'unet':
		model = UNET()
	else:
		raise ValueError('Unknown model arch type: ' + args.model_arch.lower())
		
	return model

def chooseLoss(args, option=0):
	
	loss_fn = None
	if args.model_arch.lower() == 'vdsrr' and option == 0:
		loss_fn = nn.BCEWithLogitsLoss()
	elif args.model_arch.lower() == 'unet':
		loss_fn = nn.BCEWithLogitsLoss()
	else:
		raise ValueError('Unsupported loss function')
		
	return loss_fn

def chooseOptimizer(model, args):
	
	optimizer = None
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
	elif args.optimizer == 'custom':
		pass
	else:
		raise ValueError('Unsupported optimizer: ' + args.optimizer)

	return optimizer
	

