"""
Created on Fri Oct 20 13:01:07 2017

@author: ayooshmac
"""


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')
from torchnet import meter

import pickle as pkl 

import torch 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torchvision import transforms 

class custom_model(object):
    def __init__(self, model, loss_fn):
        self.model = model
        self.original_loss_fn = loss_fn
        self.loss_fn = loss_fn
        self.losses = []
        self.losses_test = []
        self.accus = []
        self.accus_train = []
        self.vals = []
    
    def train(self, trainloader, testloader, validloader, optimizer, epochs, plot = False):
        self.losses = []
        self.losses_test = []
        self.accus = []
        self.accus_train = []
        patience = 8
        j = 0
        prev_valid_score = 0
        best_valid = 0
        
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):
                x, y = data
                x,y = Variable(x), Variable(y)
                optimizer.zero_grad()
                
                outputs = self.model(x)

                loss = self.loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                
            #implementation of early stopping 
            
            
            #logging metrics for plotting, 
            self.losses.append(self.get_loss(trainloader))
            self.losses_test.append(self.get_loss(testloader))
            self.accus.append(self.metrics_val(testloader)[0])
            self.accus_train.append(self.metrics_val(trainloader)[0])
            
            curr_valid_score = (self.metrics_val(validloader)[0])
            
            if curr_valid_score > best_valid:
                best_valid  =  curr_valid_score
            if curr_valid_score <= prev_valid_score:
                j = j + 1
            else:
                j = 0
            if (j == patience):
                #print("Epochs trained: ", epoch)
                #print(self.metrics_val(testloader)[0])
                break
                
            prev_valid_score = curr_valid_score
        return curr_valid_score
