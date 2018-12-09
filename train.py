import time
import math
import os
import os.path as osp
import argparse
import logging

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from dataset.LipClassSeg import *
from model.FuzzyNet import FuzzyNet
from model.FuzzyNet import FuzzyLayer
from trainer import Trainer 


configurations = {
    1: dict(
        max_iteration=50000,
        lr=3e-07,
        momentum=0.99,
        weight_decay=0.005,
        interval_validate=500,
    ),
    2: dict(
        max_iteration=50000,
        lr=2e-07,
        momentum=0.99,
        weight_decay=0.005,
        interval_validate=500,
    ),
    3: dict(
        max_iteration=50000,
        lr=5e-07,
        momentum=0.99,
        weight_decay=0.005,
        interval_validate=500,
    ),
    4: dict(
        max_iteration=50000,
        lr=2.0e-06,
        momentum=0.99,
        weight_decay=0.005,
        interval_validate=200,
    ),
    5: dict(
        max_iteration=20000,
        lr=3.0e-06,
        momentum=0.99,
        weight_decay=0.005,
        interval_validate=200,
    )
}

if __name__ == '__main__':
	now_time = time.asctime(time.localtime(time.time()))
	logging_path = "log/log_" + now_time.replace(" ","_") + ".txt"
	msg = "Fuzzy Lip Segmentation Traning Starting...." 
	print(msg)
	logging.basicConfig(filename=logging_path, filemode="w", level=logging.DEBUG)
	logging.info('Logging Debug .')
	logging.info(msg)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type=int, required=True)
	parser.add_argument('-c', '--config', type=int, default=1,
                            choices=configurations.keys())
	parser.add_argument('--checkpoint', help='Checkpoint path')
	args = parser.parse_args()	

	print(args)
	logging.info(args)

	gpu = args.gpu
	cfg = configurations[args.config]
	resume = None#args.resume

	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

	cuda = torch.cuda.is_available()
	logging.info("Cuda value is ")
	logging.info(cuda)

	torch.manual_seed(950624)
	if cuda:
        	torch.cuda.manual_seed(950624)

	########################################################################################
	#1 Load the dataset
        
	img_path = osp.expanduser('/home/chengguan/lip_data/DlbProcess')
	kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
	
	train_loader = torch.utils.data.DataLoader(
					LipClassSeg(root=img_path, split='train', transform=True),
					batch_size=10, shuffle=True, **kwargs)
	val_loader = torch.utils.data.DataLoader(
					LipClassSeg(root=img_path, split='val', transform=True),
					batch_size=10, shuffle=True, **kwargs)
	print(len(val_loader))
	print(len(train_loader))	
	print("Data Loaded OK.")
	########################################################################################
	#2 Load the model
	
	model = FuzzyNet(n_class=2,testing=False)
	if cuda:
		model = model.cuda()     
	print("Model Loaded OK.")
	########################################################################################
	#3 OPT
	
	optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])	

	########################################################################################
	#4 TRAIN

	out = "log/val_curve.txt."+str(args.config)
	out_model = "trained_models/FuzzyNet_config_"+str(args.config)+".model"
	start_time = time.time()
	trainer = Trainer(
			cuda=cuda,
			model=model,
			optimizer=optim,
			train_loader=train_loader,
			val_loader=val_loader,
			out=out,
			out_model=out_model,
			max_iter=cfg['max_iteration'],
			interval_validate=cfg.get('interval_validate', len(train_loader)),
			log=logging_path        
		)
	trainer.train()
	end_time = time.time()

	########################################################################################

	logging.info("Traning End.")
	logging.info("Traning Time is \n\t")
	logging.info(end_time-start_time)



