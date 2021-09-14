from __future__ import print_function
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch.nn import init
from collections import OrderedDict
import time
import shutil
import xlwt
from xlwt import Workbook 
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import random
random.seed(6)
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import torch as th
from module import validate,validate1,bin2int,weight_conversion,int2bin
from model import vgg11_bn,quan_Linear,quan_Conv2d,ResNetBasicblock,DownsampleA,CifarResNet
from attack import DES_new
import argparse

parser = argparse.ArgumentParser(description='Deep Dup A')
parser.add_argument('--iteration', type=int, default=1000, help='Attack Iterations')
parser.add_argument('--z', type=int, default=500, help='evolution z')
parser.add_argument('--batch_size',  type=int, default=256, help='input batch size for 256 default')
parser.add_argument('--probab',  type=float, default=1, help='probability of a successfull hardware AWD attack at a location')
parser.add_argument('--data', type=str, default='./cifar10', help='data path')
args = parser.parse_args()
print(args)

# datapath for the workstation
dataset_path= args.data

# ---------------------------------- parameters --------------------------------------

iteration = args.iteration ## number of attack iteration
picks = args.z # number of weights picked initially
weight_p_clk = 2 ## number of weights at each package 
shift_p_clk = 1  ## number of clock shift at each iteration 
evolution = args.z  ## number of evolution = picks = number of initial candidate chosen =z
targeted = 8  ## ignore
BATCH_SIZE =args.batch_size ## batch_size
probab = args.probab # AWD success probability $f_p$

#---------------- data loading ---------------------------------------




device=1
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dataset_path, train=True, download=True,
                       transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=False)

test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dataset_path, train=False, 
                         transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=True)



# ------------------------ model -----------------------------------
model = CifarResNet(ResNetBasicblock, 20, 10)
model=model.cuda()

criterion = torch.nn.CrossEntropyLoss()

criterion=criterion.cuda()



# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
pretrained_dict = torch.load('Resnet20_8_0.pkl')
model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(model_dict)
n=0
# update the step size before validation
for m in model.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        n=n+1
        print(m.weight.size(),n)  
        m.__reset_stepsize__()
        m.__reset_weight__()


weight_conversion(model)

validate(model, device, criterion, test_loader, 0)


# see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
model.eval()

import copy
model1=copy.deepcopy(model)
for batch_idx, (data, target) in enumerate(test_loader):
    data, target = data, target
    break
            
#----------------------------------- Attack Setup -----------------------------------------------------------------------------

attacker = DES_new(criterion, k_top=picks, w_clk=weight_p_clk, s_clk=shift_p_clk,evolution= evolution,probab=probab)
xs=[]
ys=[]
ASR=torch.zeros([iteration])
acc=torch.zeros([iteration])

# ----------------------------------- Attacking ----------------------------------------------------------------
# test case 2
for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data, target
        break
for i in range(iteration):
        print("epoch:",i+1)
        xs,ys=attacker.progressive_search(model.cuda(), data.cuda(), target.cuda(),xs,ys)
        #print(xs[i],ys[i])
        
        _,acc[i] = validate(model, device, criterion, test_loader, 0)
        if float(acc[i]) < 12.00:
            break
## finally printing out exactly how many weights different compared to the original model
i=0
for name, m in model.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            i=i+1
            j=0
            for name1, h in model1.named_modules():
                
                if isinstance(h, quan_Conv2d) or isinstance(h, quan_Linear):
                    j=j+1
                    if i==j:
                        zz=m.weight.data-h.weight.data
                        print(zz[zz!=0].size())
