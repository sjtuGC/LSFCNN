import math

import numpy as np
import torch
import torch.nn as nn
import itertools

class FuzzyLayer(nn.Module):
	def __init__(self, fuzzynum,channel):
		super(FuzzyLayer,self).__init__()
		self.n = fuzzynum
		self.channel = channel
		self.conv1 = nn.Conv2d(self.channel,1,3,padding=1)
		self.conv2 = nn.Conv2d(1,self.channel,3,padding=1)
		self.mu = nn.Parameter(torch.randn((self.channel,self.n)))
		self.sigma = nn.Parameter(torch.randn((self.channel,self.n)))
		self.bn1 = nn.BatchNorm2d(1, affine=True)
		self.bn2 = nn.BatchNorm2d(self.channel,affine=True)

	def forward(self, x):
		x = self.conv1(x)
		tmp = torch.tensor(np.zeros((x.size()[0],x.size()[1],x.size()[2],x.size()[3])),dtype = torch.float).cuda()
		for num,channel,w,h in itertools.product(range(x.size()[0]),range(x.size()[1]),range(x.size()[2]),range(x.size()[3])):
			for f in range(self.n):
				tmp[num][channel][w][h] -= ((x[num][channel][w][h]-self.mu[channel][f])/self.sigma[channel][f])**2
		fNeural = self.bn2(self.conv2(self.bn1(torch.exp(tmp))))
		return fNeural				

class FuzzyNet(nn.Module):
	def __init__(self, n_class=2, testing=False):
		super(FuzzyNet, self).__init__()

		self.fuzzy_4 = FuzzyLayer(fuzzynum=1,channel=512)
		self.fuzzy_3 = FuzzyLayer(fuzzynum=0,channel=256)
		self.fuzzy_2 = FuzzyLayer(fuzzynum=0,channel=128)
		self.fuzzy_1 = FuzzyLayer(fuzzynum=0, channel=64)

		self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
		self.relu1_1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
		self.relu1_2 = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
		# 1/2

		self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
		self.relu2_1 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
		self.relu2_2 = nn.ReLU(inplace=True)

		self.conv2_r1 = nn.Conv2d(64, 64, 1)
		self.bn2_r1   = nn.BatchNorm2d(64,affine=True)
		self.relu2_r1 = nn.ReLU(inplace=True)
		self.conv2_r2 = nn.Conv2d(64, 128, 3, padding=1)
		self.bn2_r2   = nn.BatchNorm2d(128,affine=True)
		self.relu2_r2 = nn.ReLU(inplace=True)
		self.conv2_r3 = nn.Conv2d(128,128, 1)

		self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
		# 1/4

		self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
		self.relu3_1 = nn.ReLU(inplace=True)
		self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
		self.relu3_2 = nn.ReLU(inplace=True)
		self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
		self.relu3_3 = nn.ReLU(inplace=True)

		self.conv3_r1 = nn.Conv2d(128, 128, 1)
		self.bn3_r1   = nn.BatchNorm2d(128,affine=True)
		self.relu3_r1 = nn.ReLU(inplace=True)
		self.conv3_r2 = nn.Conv2d(128, 256, 3, padding=1)
		self.bn3_r2   = nn.BatchNorm2d(256,affine=True)
		self.relu3_r2 = nn.ReLU(inplace=True)
		self.conv3_r3 = nn.Conv2d(256,256, 1)

		self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
		# 1/8

		self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
		self.relu4_1 = nn.ReLU(inplace=True)
		self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_2 = nn.ReLU(inplace=True)
		self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_3 = nn.ReLU(inplace=True)

		self.conv4_r1 = nn.Conv2d(256, 256, 1)
		self.bn4_r1   = nn.BatchNorm2d(256,affine=True)
		self.relu4_r1 = nn.ReLU(inplace=True)
		self.conv4_r2 = nn.Conv2d(256, 512, 3, padding=1)
		self.bn4_r2   = nn.BatchNorm2d(512,affine=True)
		self.relu4_r2 = nn.ReLU(inplace=True)
		self.conv4_r3 = nn.Conv2d(512,512, 1)
		
		self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
		# 1/16
		
		self.deconv1 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(2,2),stride=(2,2),bias=False)
		self.deconv2 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(2,2),stride=(2,2),bias=False)
		self.deconv3 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(2,2),stride=(2,2),bias=False)
		self.deconv4 = nn.ConvTranspose2d(in_channels=64,out_channels=2,kernel_size=(2,2),stride=(2,2),bias=False)

		self.fbn1 = nn.BatchNorm2d(64, affine=True)
		self.fbn2 = nn.BatchNorm2d(128, affine=True)
		self.fbn3 = nn.BatchNorm2d(256, affine=True)
		self.fbn4 = nn.BatchNorm2d(512, affine=True)

		self.bn1 = nn.BatchNorm2d(512, affine=True)
		self.bn2 = nn.BatchNorm2d(256, affine=True)
		self.bn3 = nn.BatchNorm2d(128, affine=True)
		self.bn4 = nn.BatchNorm2d(64, affine=True)

		self.testing = testing 
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				#m.weight.data.zero_()
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.ConvTranspose2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels*m.in_channels
				m.weight.data.normal_(0,math.sqrt(2. / n))
				if m.bias is not None:
                                        m.bias.data.zero_()
			if isinstance(m,nn.Linear):
				m.weight.data.normal_(0, 1)

		for param in self.parameters():
			param.requires_grad = True

	def forward(self, x):
		h = x
		h = self.relu1_1(self.conv1_1(h))
		h = self.relu1_2(self.conv1_2(h))
		h = self.pool1(h)
		c1 = h
		f1 = self.fbn1(self.fuzzy_1(c1))+c1

		g = h
		g = self.relu2_r1(self.bn2_r1(self.conv2_r1(g)))
		g = self.relu2_r2(self.bn2_r2(self.conv2_r2(g)))
		g = self.conv2_r3(g)

		h = self.relu2_1(self.conv2_1(h))
		h = self.relu2_2(self.conv2_2(h))
		h = self.pool2(h+g)
		c2 = h
		f2 = self.fbn2(self.fuzzy_2(c2))+c2

		g = h
		g = self.relu3_r1(self.bn3_r1(self.conv3_r1(g)))
		g = self.relu3_r2(self.bn3_r2(self.conv3_r2(g)))
		g = self.conv3_r3(g)

		h = self.relu3_1(self.conv3_1(h))
		h = self.relu3_2(self.conv3_2(h))
		h = self.relu3_3(self.conv3_3(h))
		h = self.pool3(h+g)
		c3 = h
		f3 = self.fbn3(self.fuzzy_3(c3))+c3

		g = h
		g = self.relu4_r1(self.bn4_r1(self.conv4_r1(g)))
		g = self.relu4_r2(self.bn4_r2(self.conv4_r2(g)))
		g = self.conv4_r3(g)
		h = self.relu4_1(self.conv4_1(h))
		h = self.relu4_2(self.conv4_2(h))
		h = self.relu4_3(self.conv4_3(h))
		h = self.pool4(h+g)
		c4 = h
		f4 = self.fbn4(self.fuzzy_4(c4))
		h = self.bn1(f4+h)
		h = self.bn2(self.deconv1(h)+f3)
		h = self.bn3(self.deconv2(h)+f2)
		h = self.bn4(self.deconv3(h)+f1)
		h = self.deconv4(h)
		return h
