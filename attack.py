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
from model import quan_Linear,quan_Conv2d

class DES_old(object):
    def __init__(self, criterion, k_top=50, w_clk=1, s_clk=1, evolution=1000):

        self.criterion = criterion.cuda()
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        ## number of weights per clock or group
        self.N=w_clk
        ## number of wweight shift at one iteration
        self.S=s_clk
        ## tracking number of pool of weights
        self.k_top = k_top
        self.total=0
        ##evolution z
        self.epoch=evolution

    def shift(self, m,f_index):
        
        # size of the layer
        self.total=m.weight.detach().view(-1).size()[0] ## size of the entire layer
        ranks=self.k_top ## rank size can be different if needed to speed up
        
        ranging=self.N ## number of byte transferred
        
        params=m.weight.data.detach().clone() ##  weights
        param2= m.weight.data.detach().clone() 
        param_flat=params.view(-1)
        
        param_flat = torch.flip(param_flat,[0])
        for y in range(self.S*ranging): 
            param_flat[f_index+y]=param_flat[f_index+ranging+y]  ## shifting the values 
        param_flat = torch.flip(param_flat,[0])
        params = param_flat.view(params.size())
        #w=param2.view(-1)-params.view(-1) ## just to check how many weights changes

        param_flipped=params.detach().clone() ## copying the parameters
        

        return param_flipped

    def shift2(self, m,f_index):
        
        # size of the layer
        self.total=m.weight.detach().view(-1).size()[0] ## size of the entire layer
        ranks=self.k_top ## rank size can be different if needed to speed up
        
        ranging=self.N ## number of byte transferred
        
        params=m.weight.data.detach().clone() ##  weights
        param2= m.weight.data.detach().clone() 
        param_flat=params.view(-1)
        
        #k1=params.size()[2]
        #k0=params.size()[3]
        #ins=params.size()[1]
        #outs=params.size()[0]
       
        
        
        
        #print(f_index)
        #print(self.total-f_index-1)
        
        param_flat = torch.flip(param_flat,[0])
        for y in range(self.S*ranging):
            print(param_flat[f_index+y],param_flat[f_index+ranging+y])    
            param_flat[f_index+y]=param_flat[f_index+ranging+y]  ## shifting the values 
        param_flat = torch.flip(param_flat,[0])
        params = param_flat.view(params.size())
        param_flipped=params.detach().clone() ## copying the parameters
        return param_flipped 
    def mutation(self,model,data,target,obj_func,x,y,layers,y_max,mutation=0):
        train_indices = torch.from_numpy(np.random.choice(self.k_top, size=(5), replace=False)) 
        x,y=x.float(),y.float()
        x_norm=x/layers ## normalize
        y_norm=torch.div(y,y_max.float())  ## Normalize
        x,y=x.int(),y.int()
        #print(x,y)
        F_1 = torch.clamp(torch.rand(1),0,1) 
        F_2 = torch.clamp(torch.rand(1),0,1)  
        F_3 = torch.clamp(torch.rand(1),0,1)
         
        ## mutant vector
        if mutation == 3:
           _,indx = obj_func.topk(self.k_top)
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[[indx[0]]]-x_norm[[indx[-1]]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[[indx[0]]]-y_norm[[indx[-1]]])
        if mutation == 0:
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[train_indices[1]]-x_norm[train_indices[2]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[train_indices[1]]-y_norm[train_indices[2]])
        if mutation == 1:
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[train_indices[1]]-x_norm[train_indices[2]]) + F_1 * (x_norm[train_indices[3]]-x_norm[train_indices[4]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[train_indices[1]]-y_norm[train_indices[2]]) + F_1 * (y_norm[train_indices[3]]-y_norm[train_indices[4]])
        #print(mut_x,mut_y)
        if mutation == 2:
           _,indx = obj_func.topk(self.k_top)
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[[indx[0]]]-x_norm[train_indices[0]]) + F_1 * (x_norm[train_indices[1]]-x_norm[train_indices[2]]) + F_1 * (x_norm[train_indices[3]]-x_norm[train_indices[4]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[[indx[0]]]-y_norm[train_indices[0]]) + F_1 * (y_norm[train_indices[1]]-y_norm[train_indices[2]]) +  F_1 * (y_norm[train_indices[3]]-y_norm[train_indices[4]])  
        #print(mut_x,mut_y)
        skip = 0
        ## they should be within this range [0,1] or just skip this evolution       
        if mut_x>1 or  mut_x<0:
           skip = 1 
        if mut_y>1 or mut_y <0:
           skip = 1
        if skip == 0:    
            ## denormalization of x
            mut_x=int(mut_x*layers)
            n=0
            for m in model.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                    
                    if n == mut_x:
                       mut_y= (mut_y*(m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S)) # denormalization of y
                    n=n+1
            mut_y = int(mut_y)
            #print(mut_x)
            obj_new = 100000
            n=0
            for m in model.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                    
                    if n == mut_x:
                        clean_weight = m.weight.data.detach().clone()
                        attack_weight=self.shift( m,mut_y)  # perform shift at x,y
                        m.weight.data = attack_weight
                        output=model(data)
                        obj_new=self.criterion(output,target).item() # update loss
                        m.weight.data= clean_weight # recover the shift
                    n=n+1

            # just check with the worst one        
            _,indx= obj_func.topk(self.k_top)
            #print(obj_func[indx[0]])
            if obj_func[indx[0]]< obj_new: # if current one is better than the worst then replace it or move on
                #print(mutation)
                obj_func[indx[0]] = obj_new
                x[indx[0]]=mut_x
                y[indx[0]]=mut_y
        return obj_func, x, y


    def progressive_search(self, model, data, target,xs,ys):
      
        # Note that, attack has to be done in evaluation model due to batch-norm.
        model.eval()
        #target[:] = targets ## select the target class
       
        # 2. zero out the grads first, then get the grads
        n=0
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
               n=n+1
        max_weight=torch.zeros([n+1])
        layers=n
        print(layers)
        n=0
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
               n=n+1
               max_weight[n]=m.weight.data.detach().max()
        
        # 3. for each layer flip #bits = self.bits2fli
        obj_func=torch.zeros([self.k_top])
        x=torch.randint(0, layers, ([self.k_top]))
        #print(x)
        y=torch.randint(0, layers, ([self.k_top]))
        y_max=torch.randint(0, layers, ([self.k_top])).float()
        for i in range(self.epoch):
            
            # iterate all the quantized conv and linear layer for i= 0 to initialize the loss functions
            
            if i == 0:
                for k in range(self.k_top ):
                    n=0
                    for m in model.modules():
                        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                            
                            if n == x[k]:
                                clean_weight = m.weight.data.detach().clone()
                                y[k]=torch.from_numpy(np.random.choice(int(m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S), size=(1), replace=False)) 
                                y_max[k]= m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S
                                attack_weight=self.shift( m,y[k])
                                m.weight.data = attack_weight
                                output=model(data)
                                obj_func[k]=self.criterion(output,target).item()
                                m.weight.data = clean_weight
                                #print(obj_func[k])
                            n=n+1
                x_init=x.detach().clone()
                #print(obj_func)
            for z in range(4):
                obj_func ,x ,y = self.mutation(model,data,target,obj_func,x,y,layers,y_max,mutation=z)
        #print(x.long()-x_init.long())


        ## in the end I have a list of   
        count = 1
        number = 0
        _,indx = obj_func.topk(self.k_top)
        while count == 1:
            check=xs.view(1, -1).eq((x[indx[number]]).float().view(-1, 1)).sum(0).sum() ## checking if current x matches any prev xs
            if int(check) ==1:
                check2=ys.view(1, -1).eq((y[indx[number]]).float().view(-1, 1)).sum(0).sum() ## checking if current y matches prev ys
                if int(check2) == 1:
                    number=number+1
                else: 
                    count=0
            else:
                count=0

        ## after the check number will indicate the index where we perform the shift

        n=0
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                #print(n,name)
               if n==x[indx[number]]:
                #print(name, self.loss.item(), loss_max)
                   attack_weight = self.shift2(m,y[indx[number]])
                   m.weight.data = attack_weight
               n=n+1

        print( x[indx[number]],y[indx[number]],number)
        
        return x[indx[number]],y[indx[number]]



class DES_new(object):
    def __init__(self, criterion, k_top=50, w_clk=1, s_clk=1, evolution=1000):
        
        self.criterion = criterion.cuda() 
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        ## number of weights per clock or group
        self.N=w_clk
        ## number of wweight shift at one time
        self.S=s_clk
        ## tracking number of pool of weights
        self.k_top = k_top
        
        self.total=0
        ##evolution z
        self.epoch=evolution

    def shift(self, m,f_index):
        ''' performs the s_clk number of shift starting at index f_index given a layers weights m'''

        # size of the layer
        self.total=m.weight.detach().view(-1).size()[0] ## size of the entire layer
        ranks=self.k_top ## initial pool 
        
        ranging=self.N ## number of weight in each clock
        
        params=m.weight.data.detach().clone() ##  weights
        param2= m.weight.data.detach().clone() ## keeping another copy for record
        
        
        
        
        param_flat=params.view(-1) ## 1D flattended
        param_new1 = param_flat[200].detach().clone()
        param_new2 = param_flat[400] .detach().clone()
        param_flat[200]=0
        param_flat[400]=0
        param_flat = torch.flip(param_flat,[0]) ## flipping to make right shift real AWD genrates right shift thats why
        for y in range(self.S*ranging): 
            param_flat[f_index+y]=param_flat[f_index+ranging+y]  ## shifting the values 
        param_flat = torch.flip(param_flat,[0]) 
        param_flat[200]=param_new1
        param_flat[400]=param_new2
        param2 = param_flat.view(params.size())  # putting it back to the original matrix
        
       

        param_flipped=param2.detach().clone() ## copying the parameters

        return param_flipped

    def shift2(self, m,f_index):
        ''' performs the s_clk number of shift starting at index f_index given a layers weights m'''
        ## same as before not necessary 
        self.total=m.weight.detach().view(-1).size()[0] ## size of the entire layer
        ranks=self.k_top ## rank size can be different if needed to speed up
        
        ranging=self.N ## number of byte transferred
        
        params=m.weight.data.detach().clone() ##  weights
        param2= m.weight.data.detach().clone() 

        

        param_flat=params.view(-1)
        param_new1 = params.view(-1)[200].detach().clone()
        param_new2 = params.view(-1)[400] .detach().clone()
        param_flat[200]=0
        param_flat[400]=0
        param_flat = torch.flip(param_flat,[0])
        for y in range(self.S*ranging):
            print("Old value, new value:")
            print(param_flat[f_index+y],param_flat[f_index+ranging+y])    
            param_flat[f_index+y]=param_flat[f_index+ranging+y]  ## shifting the values 
        param_flat = torch.flip(param_flat,[0])

        param_flat[200]=param_new1
        param_flat[400]=param_new2

        param2 = param_flat.view(params.size())
       
        param_flipped=param2.detach().clone() ## copying the parameters
        return param_flipped 


    def mutation(self,model,data,target,obj_func,x,y,layers,y_max,h,mutation=0):
        ''' this function performs the mutation step 
            model : network architecture
            data : test data
            target: label of the data
            obj_func: initial population mutation function values
            x: current x
            y: current y
            y_max: total weights at layer x
            mutation : which mutation strategy to use
            h: evolution number
        '''
        # random numbers to perform mutation
        train_indices = torch.from_numpy(np.random.choice(self.k_top, size=(5), replace=False)) 
        
        # normalization step
        x_norm=torch.clamp(x.float()/layers,0,1) ## normalize
        y_norm=torch.clamp(torch.div(y.float(),y_max.float()),0,1)  ## Normalize
        x,y=x.int(),y.int()
        
        # generating three alphas
        F_1 = torch.clamp(torch.rand(1),0.3,1) 
        F_2 = torch.clamp(torch.rand(1),0.3,1)  
        F_3 = torch.clamp(torch.rand(1),0.3,1)
         
        ## four mutantation strategy 
        if mutation == 3:
           _,indx = obj_func.topk(self.k_top)
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[[indx[0]]]-x_norm[[indx[-1]]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[[indx[0]]]-y_norm[[indx[-1]]])

        if mutation == 0:
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[train_indices[1]]-x_norm[train_indices[2]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[train_indices[1]]-y_norm[train_indices[2]])

        if mutation == 1:
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[train_indices[1]]-x_norm[train_indices[2]]) + F_2 * (x_norm[train_indices[3]]-x_norm[train_indices[4]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[train_indices[1]]-y_norm[train_indices[2]]) + F_2 * (y_norm[train_indices[3]]-y_norm[train_indices[4]])

        if mutation == 2:
           _,indx = obj_func.topk(self.k_top)
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[[indx[0]]]-x_norm[train_indices[0]]) + F_2 * (x_norm[train_indices[1]]-x_norm[train_indices[2]]) + F_3 * (x_norm[train_indices[3]]-x_norm[train_indices[4]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[[indx[0]]]-y_norm[train_indices[0]]) + F_2 * (y_norm[train_indices[1]]-y_norm[train_indices[2]]) +  F_3 * (y_norm[train_indices[3]]-y_norm[train_indices[4]])  
        
        # ignore the skip varibale for corssover case
        skip = 0

        ## mutatant should be within this range [0,1] or just replace with the parent features      
        if mut_x>1 or  mut_x<0:
           mut_x = x_norm[h]
        if mut_y>1 or mut_y <0:
           mut_y = y_norm[h]
        if skip == 0:

            ## denormalization of x and y
            mut_x=int(mut_x*layers)
            n=0
            for m in model.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                    
                    if n == mut_x:
                       
                       mut_y= (mut_y* m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S) # denormalization of y
                       
                    n=n+1
            mut_y = int(mut_y)

            #compute the new objective for the new mutant vector
            obj_new = 100000
            n=0
            for m in model.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                    
                    if n == mut_x:
                        clean_weight = m.weight.data.detach().clone()
                        attack_weight=self.shift( m,mut_y)
                        m.weight.data = attack_weight
                        output=model(data)
                        obj_new=self.criterion(output,target).item() ## new mutation function evaluation at mut_x and mut_y
                        m.weight.data= clean_weight
                    n=n+1

            
            #compare with current population if causes more damage then replace
            if obj_func[h] < obj_new:
                #print(mutation)
                obj_func[h] = obj_new
                x[h]=mut_x
                y[h]=mut_y
        return obj_func, x, y


    def progressive_search(self, model, data, target,xs,ys):
      
        # set the model to evaluation mode
        model.eval()
       
       
        # calculating total number of layers in the model we just attack convolution and linear layers
        n=0
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
               n=n+1
        max_weight=torch.zeros([n+1])
        layers=n
        print("Number of layers:" ,layers)
        
        # 3. setting up initial objective function,x anb y
        obj_func=torch.zeros([self.k_top])
        x=torch.randint(0, layers, ([self.k_top]))
        #print(x)
        y=torch.randint(0, layers, ([self.k_top]))
        y_max=torch.randint(0, layers, ([self.k_top])).float()

        ## start the evolution
        for i in range(self.epoch):
            # only calculate the initial population objective for iteration 0
            if i == 0:
                for k in range(self.k_top ):
                    n=0
                    for m in model.modules():
                        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                            
                            if n == x[k]:
                                clean_weight = m.weight.data.detach().clone()
                                y[k]=torch.from_numpy(np.random.choice(int(m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S), size=(1), replace=False)) 
                                y_max[k]= m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S ## for each y value corresponding maximum y_max
                                attack_weight=self.shift( m,y[k]) # attack 
                                m.weight.data = attack_weight
                                output=model(data)
                                obj_func[k]=self.criterion(output,target).item() ## evaluate fitness function
                                m.weight.data = clean_weight ## recover the weights
                                #print(obj_func[k])
                            n=n+1
            
            #perfomr four mutation strategy for each population candidate
            for z in range(4):
                obj_func ,x ,y = self.mutation(model,data,target,obj_func,x,y,layers,y_max, i, mutation=z)

        ## This part checks if any previous shift were done at the best objective function index
        count = 0
        number = 0
        _,indx = obj_func.topk(self.k_top)
        
        ## This part checks if any previous shift were done at the best objective function index
        for k in range(indx.size()[0]):
            for i in range(len(xs)):
                if (x[indx[number]],y[indx[number]]) == ( xs[i], ys[i]):
                    count = 1
            if count == 1:
                number =  number +1
                count= 0
            else:
                break


                
        #This part checks if any previous shift were done at the best objective function index -1
        for k in range(indx.size()[0]):
            for i in range(len(xs)):
                if (x[indx[number]],y[indx[number]]) == ( xs[i], ys[i]+1):
                    count = 1
            if count == 1:
                number =  number +1
                count=0
            else:
                break
        #This part checks if any previous shift were done at the best objective function index +1 since the attack effects two weights
        for k in range(indx.size()[0]):
            for i in range(len(xs)):
                if (x[indx[number]],y[indx[number]]) == ( xs[i], ys[i]-1):
                    count = 1
            if count == 1:
                number =  number +1
                count=0
            else:
                break
        ## the reason we need to do that because lets assume attack at index 2 [1,2,3,4] after a shift [1,1,2,4] so basically we can not attack [. X X X] (2+1) and (2-1) anymore.
                
        print(number)
        ## after the check 'number' will indicate the index where we perform the shift (x[indx[number]],y[indx[number]])
        xs.append(x[indx[number]])
        ys.append(y[indx[number]])
        n=0
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                #print(n,name)
               if n==x[indx[number]]:
                #print(name, self.loss.item(), loss_max)
                   attack_weight = self.shift2(m,y[indx[number]])
                   m.weight.data = attack_weight
               n=n+1

        print("Layer numer, Index Number: ", x[indx[number]],y[indx[number]])
        
        return xs,ys