import os
import sys
import argparse
import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils import data

from lib.utils import *
from tool.tool import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_normalization = False

# output directory
root_dir = '/home/kuowei/Downloads/'
checkpoint_dir = './checkpoint/'
output_log_dir = './log/'
tb_log_dir = './tb_log/'
vis_output_path = './vis_output/'

# prediction data number
prediction_set = 'val' #['train', 'val', 'test', 'ext']
subject_num = '1_20' 
patch_num = '88'
case = 'whole' #['patch', 'whole']

prediction_data = (prediction_set, subject_num, patch_num, case)

def main(args):

	device = torch.device('cuda:0' if not args.no_cuda else 'cpu')

	if args.mode == 'train':

		## experiment name
		exp_name = args.model_arch + args.name

		## tensorboard log
		tb_writer = SummaryWriter(tb_log_dir + exp_name)

		## data augmentation
		# Todo :design which type of data aug
		data_aug = None

		## load dataset
		print('Load dataset...')
		train_loader, val_loader = prepareDataset(args, root_dir, data_aug, normalize=data_normalization)
		print(args.dataset.lower() + ' dataset loaded.')

		## load model
		model = chooseModel(args)
		model.to(device)
		print(args.model_arch + ' loaded.')

		## parallel model
		if args.gpu_num > 1:
			model = nn.DataParallel(model)

		## loss function and optimizer
		loss_fn = chooseLoss(args, 0)	

		optimizer = chooseOptimizer(model, args)

		## initailize statistic result
		start_epoch = 0
		best_per_index = 1000
		total_tb_it = 0

		## resume training
		# Todo

		for epoch in range(start_epoch, args.num_epoch):

			total_tb_it = train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it)
			mse_index = validate(device, model, val_loader, epoch, loss_fn, tb_writer)

			state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}

			if mse_index <= best_per_index:
				best_per_index = mse_index
				best_name = checkpoint_dir + exp_name +'_Bmodel.pkl'
				torch.save(state, best_name)
			else:
				name = checkpoint_dir + exp_name +'_Emodel.pkl'
				torch.save(state, name)

		tb_writer.close()

	elif args.mode == 'predict':

		print('Load data...')
		data_loader = loadData(args, root_dir, prediction_data, normalize=data_normalization)

		## load model
		model = chooseModel(args)
		model.load_state_dict(torch.load(args.resume_file, map_location=device)['model_state'])
		model.to(device)
		print(args.model_arch + ' loaded.')

		## parallel model
		if args.gpu_num > 1:
			model = nn.DataParallel(model)

		model_name = args.resume_file.split('/')[-1].split('.')[0]
		print(model_name)

		## loss function
		loss_fn = chooseLoss(args, 0)

		predict(args, device, model, data_loader, loss_fn, model_name)
	else:
		raise Exception('Unrecognized mode.')


def train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it):

	print_freq = (len(train_loader.dataset) // args.batch_size) // 3

	model.train()

	for batch_count, (input_data, gt_data, input_name) in enumerate(train_loader):

		#cuda
		input_data = input_data.to(device, dtype=torch.float)
		gt_data = gt_data.to(device, dtype=torch.float)

		optimizer.zero_grad()

		output_data = model(input_data)

		loss = loss_fn(output_data, gt_data)

		loss.backward()

		optimizer.step()

		per_loss = loss.item()

		tb_writer.add_scalar('train/overall_loss', per_loss, total_tb_it)
		total_tb_it += 1

		if batch_count%print_freq == 0:
			print('Epoch [%d/%d] Loss: %.8f' %(epoch, args.num_epoch, per_loss))

	return total_tb_it

def validate(device, model, val_loader, epoch, loss_fn, tb_writer):

	model.eval()

	tb_loss = 0

	with torch.no_grad():

		for batch_count, (input_data, gt_data, input_name) in enumerate(val_loader):

			#cuda
			input_data = input_data.to(device, dtype=torch.float)
			gt_data = gt_data.to(device, dtype=torch.float)

			output_data = model(input_data)
			
			loss = loss_fn(output_data, gt_data)

			tb_loss += loss.item()

		avg_tb_loss = tb_loss / len(val_loader.dataset)
		print('##Validate loss: %.8f' %(avg_tb_loss))

		tb_writer.add_scalar('val/overall_loss', avg_tb_loss, epoch)

	return avg_tb_loss

def predict(args, device, model, data_loader, loss_fn, model_name):

	nifti_path = root_dir + 'img_seg/dataset/original_data/'

	model.eval()

	if case == 'patch':
		with torch.no_grad():
			for batch_count, (input_data, gt_data, input_name) in enumerate(data_loader):

				#cuda
				input_data = input_data.to(device, dtype=torch.float)
				gt_data = gt_data.to(device, dtype=torch.float)

				output_data = model(input_data)

				output_data = torch.squeeze(torch.sigmoid(output_data), 0).cpu().numpy()
				if not args.no_save:
					
					data_path = nifti_path + 'image/' + input_name[0].split('_image')[0] + '/' +  input_name[0].split('_image')[0] + '.hdr'

					if not os.path.exists(vis_output_path + model_name):
						os.makedirs(vis_output_path + model_name)

					save_name = vis_output_path + model_name + '/' + input_name[0].split('_image')[0] + '_pred'

					seg_display(output_data, data_path, out_name=save_name)

	elif case == 'whole':
		with torch.no_grad():
			for batch_count, (input_data, gt_data, input_name) in enumerate(data_loader):

				#cuda
				input_data = input_data.to(device, dtype=torch.float)
				gt_data = gt_data.to(device, dtype=torch.float)

				output_data = model(input_data)

				#output_data = torch.squeeze(torch.sigmoid(gt_data), 0).cpu().numpy()
				output_data = torch.squeeze(input_data, 0).cpu().numpy()

				if not args.no_save:
					
					data_path = nifti_path + 'image/' + input_name[0].split('_image')[0] + '/' +  input_name[0].split('_image')[0] + '.hdr'

					if not os.path.exists(vis_output_path + model_name):
						os.makedirs(vis_output_path + model_name)

					save_name = vis_output_path + model_name + '/' + input_name[0].split('_image')[0] + '_pred'

					seg_display(output_data, data_path, out_name=save_name)

			
parser = argparse.ArgumentParser(description='Medical Image Segmentation')
parser.add_argument('--mode', default='train', choices=['train', 'predict'], help='operation mode: train or predict (default: train)')
parser.add_argument('--name', type=str, default='_test', help='the name of exp')
parser.add_argument('--dataset', default='seg', choices=['seg'], help='dataset to use (default: seg)')
parser.add_argument('--gpu_num', default=1, type=int, choices=[1, 2, 3, 4], help='number of gpu (default: 1)')
parser.add_argument('--model_arch', default='vdsrr', choices=['unet', 'vdsrr'], help='network model (default: unet)')
parser.add_argument('--num_epoch', default=200, type=int, metavar='N', help='number of total epochs to run (default: 1000)')
parser.add_argument('--batch_size', type=int, default=20, help='batch size (default: 10)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (default: 1e-2)')
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'], help='optimizer to use (default: adam)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no_save', action='store_true', default=False, help='disables saving tensors')
parser.add_argument('--resume_file', type=str, default='./checkpoint/vdsr_4xus_exp00_18_64_bs_2_lr_1e-5_test40_model.pkl', help='the checkpoint file to resume from')

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
