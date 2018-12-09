##############################################################
#GC 
#in SJTU
#2018/08/07
##############################################################

import torch
from torch.utils import data
import collections

import os.path as osp
import numpy as np
import PIL.Image
from PIL import Image
import scipy.io


class ClassSegBase(data.Dataset):
	class_names = np.array([
		'background',
		'lip'])
	mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

	def __init__(self,img_path,split="train",transform=True):
		self.img_path = img_path
		self.split = split
		self._transform = transform
		
		dataset_dir = self.img_path
		self.files = collections.defaultdict(list)
		for split in ['train', 'val']:
			imgsets_file = osp.join(dataset_dir, 'ImageSets/%s.txt' % split)
			for did in open(imgsets_file):
				did = did.strip()
				img_file = osp.join(dataset_dir, 'image/%s.jpg' % did)
				lbl_file = osp.join(dataset_dir, 'annotation/%s.jpg' % did)
				self.files[split].append({
							'img': img_file,
							'lbl': lbl_file,
							})
	def __len__(self):
		return len(self.files[self.split])

	def __getitem__(self, index):
		data_file = self.files[self.split][index]
		img_file = data_file['img']
		img_file = img_file + '.224*112'
		img = PIL.Image.open(img_file)
		#img = img.resize((224,112), Image.ANTIALIAS)
		img = np.array(img, dtype=np.uint8)

		lbl_file = data_file['lbl']
		lbl_file = lbl_file + '.224*112'
		lbl = PIL.Image.open(lbl_file)
		#lbl = lbl.resize((224,112), Image.ANTIALIAS)
		lbl = np.array(lbl, dtype=np.uint8)
		if self._transform:
			return self.transform(img, lbl)
		else:
			return img, lbl

	def transform(self, img, lbl):
		img = img[:, :, ::-1] 
		img = img.astype(np.float64)
		img -= self.mean_bgr
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).float()
		mask = lbl>0
		lbl[mask] = 1
		lbl = torch.from_numpy(lbl).long()
		return img, lbl

	def untransform(self, img, lbl):
		img = img.numpy()
		img = img.transpose(1, 2, 0)
		img += self.mean_bgr
		img = img.astype(np.uint8)
		img = img[:, :, ::-1]
		lbl = lbl.numpy()
		return img, lbl

class LipClassSeg(ClassSegBase):

	def __init__(self, root, split='train', transform=True):
		super(LipClassSeg, self).__init__(root, split=split, transform=transform)	
		pass
