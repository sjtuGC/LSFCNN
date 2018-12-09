import argparse
import os
import os.path as osp
import cv2

from PIL import Image
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import tqdm

from model.FuzzyNet import FuzzyNet
from model.FuzzyNet import FuzzyLayer
from dataset.LipClassSeg import *


def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
                n_class * label_true[mask].astype(int) +
                label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

def cross_entropy2d(input, target, weight=None, size_average=True):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        #print(log_p)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
        if loss is None:
                print("ERROR")
        if size_average:
                loss /= mask.data.sum()
        return loss


def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-c', '--config',type=int, default=0)
    args = parser.parse_args()

    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = "trained_models/FuzzyNet_config_"+str(args.config)+".model"

    img_path = osp.expanduser('/home/chengguan/lip_data/DlbProcess')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}


    val_loader = torch.utils.data.DataLoader(
                                        LipClassSeg(root=img_path, split='val', transform=True),
                                        batch_size=1, shuffle=False, **kwargs)

    n_class = len(val_loader.dataset.class_names)
    print(len(val_loader))
    print("Data Loaded OK.")


    print('==> Loading %s class model file: %s' %
          (2, model_file))

    model = FuzzyNet(n_class=2,testing=True)
    model.load_state_dict(torch.load(model_file))
    print(model)
    model = model.cuda()
    model.eval()

    print('Loaded Model Successfully! \n FuzzyNet by GC testing')
    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        score = model(data)
        score_cpu = score.data.cpu()
        visualizations.append(F.softmax(score_cpu,dim=1).numpy()[:,1,:,:])
        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
    metrics = label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\Accuracy:       {0}
              Accuracy Class: {1}
              Mean IU:        {2}
              FWAV Accuracy:  {3}'''.format(*metrics))

    print(np.shape(visualizations))
    print(np.shape(label_preds))
    results_folder = "results/"
    val_num = len(val_loader)
    for i in range(val_num):
        now_im = label_preds[i]
        img = np.asarray(now_im)
        img = img*100
        heat = np.asarray(visualizations[i][0])
        heat = heat*100
        cv2.imwrite(results_folder+"result_"+str(i+1)+".png", img)
        cv2.imwrite(results_folder+"heatmap/result_"+str(i+1)+"_heat.png",heat)


if __name__ == '__main__':
    main()
