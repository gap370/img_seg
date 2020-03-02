
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNET(nn.Module):
	def __init__(self, in_channel=1, out_channel=1):
		super(UNET, self).__init__()

		self.in_channel = in_channel
		self.out_channel = out_channel

		self.conv11 = nn.Sequential(nn.Conv3d(self.in_channel, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
		self.conv12 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

		self.maxpool2m = nn.MaxPool3d(2)
		self.conv21 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
		self.conv22 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

		self.maxpool3m = nn.MaxPool3d(2)
		self.conv31 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
		self.conv32 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))

		self.maxpool4m = nn.MaxPool3d(2)
		self.conv41 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
		self.conv42 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))

		self.maxpool5m = nn.MaxPool3d(2)
		self.conv51 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=5, padding=2), nn.BatchNorm3d(512), nn.ReLU(inplace=True))
		self.conv52 = nn.Sequential(nn.Conv3d(512, 512, kernel_size=5, padding=2), nn.BatchNorm3d(512), nn.ReLU(inplace=True))

		self.deconv61 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, padding=0)
		self.conv62 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
		self.conv63 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))

		self.deconv71 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0)
		self.conv72 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
		self.conv73 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))

		self.deconv81 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
		self.conv82 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
		self.conv83 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

		self.deconv91 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
		self.conv92 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
		self.conv93 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

		self.conv101 = nn.Conv3d(32, 1, kernel_size=1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		x1 = self.conv12(self.conv11(x))
		x2 = self.conv22(self.conv21(self.maxpool2m(x1)))
		x3 = self.conv32(self.conv31(self.maxpool3m(x2)))
		x4 = self.conv42(self.conv41(self.maxpool4m(x3)))
		x5 = self.conv52(self.conv51(self.maxpool5m(x4)))
	
		x = self.conv63(self.conv62(torch.cat((self.deconv61(x5), x4), dim=1)))	
		x = self.conv73(self.conv72(torch.cat((self.deconv71(x), x3), dim=1)))
		x = self.conv83(self.conv82(torch.cat((self.deconv81(x), x2), dim=1)))
		x = self.conv93(self.conv92(torch.cat((self.deconv91(x), x1), dim=1)))

		out = self.conv101(x)

		return out
		
