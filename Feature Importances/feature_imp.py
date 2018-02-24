 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:30:18 2017

@author: ayooshmac
"""

#to check the feature importances 

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')
from torchnet import meter
from tabulate import tabulate

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import pickle as pkl 
from custom_model import *
from loaders import *


import torch 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torchvision import transforms

a = open("../data/datasets", "rb")
datasets = pkl.load(a)

combined = pd.concat(datasets)

#print(len(combined.columns))

cols = datasets[0].columns[:-1]

best_all_accu = 96.3235

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 2/float(12))
        m.bias.data.normal_(0, 2/float(12))

def trials(model, params, dataloaders, num_trials):
    
    net_accuracy = 0
    net_AUROC = 0
    net_diff = 0 
    net_loss = 0
    trainloader, testloader, validloader = dataloaders
    
    
    for num in range(num_trials):
        a = custom_model(model, loss_fn)  
        a.model.apply(init_weights)
        a.train(trainloader, testloader, validloader, optimizer, 50, plot = False)
        accuracy, ct, auc, cm = a.metrics_val(testloader)
        test_train_diff = a.metrics_val(trainloader)[0] - accuracy
        net_diff += test_train_diff
        net_accuracy += accuracy
        net_AUROC += auc 
        net_loss  += a.get_loss(testloader)
        print (a.get_loss(testloader))
        
    net_accuracy = round(net_accuracy/num_trials, 4)
    net_AUROC = round(net_AUROC/num_trials, 4)
    net_diff = round(net_diff/num_trials, 4)
    net_loss = round(net_loss/num_trials, 4)
    
    return (net_accuracy, net_AUROC,cm, net_diff, net_loss)


D_in, H, D_out = combined.shape[1] - 2, 30 , 2


model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)

lr = 0.005
loss_fn = nn.CrossEntropyLoss()
wd = 0.01
optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 2/float(12))
        m.bias.data.normal_(0, 2/float(12))


all_loss  = 0.0123
params = (lr, loss_fn, wd, optimizer)
ftr_imps = dict.fromkeys(cols)
for x in cols:
    comb_drop = combined.drop(x, axis = 1)
    train, test, valid = get_partitions(comb_drop,[0.6, 0.2, 0.2])
    datasets = train, test, valid
    dataloaders = get_dataloaders(datasets, tran, batch_size = 30)
    ftr_accu, ftr_auc, ftr_cm, ftr_diff, net_loss = trials(model, params, dataloaders, 5)
    print ("Accuracy after dropping feature", x)
    print ("Accuracy:", ftr_accu)
    print("AUC:", ftr_auc)
    print(ftr_cm)
    percent =  ((all_loss - net_loss)*100/(all_loss))
    print("Difference:", percent)
    ftr_imps[x] = (percent, ftr_accu, ftr_auc)
    

file = open("feature_importances_leave_one_out", "wb")
pkl.dump(ftr_imps, file)

fs = sorted(ftr_imps, key=ftr_imps.get, reverse = True)
li = [(x, round(ftr_imps[x][0],4),round(ftr_imps[x][1],4) ) for x in fs]

fle = open("feature_importances_leave_one_out_table", "w") 
li = [("Feature", "Difference from \nCombined Accuracy", "Accuracy")] + li
print (tabulate(li, headers = "firstrow"), file = fle)
file.close()
assert False

tran = transforms.Compose([transforms.ToTensor()])
file= open("newf", "rb")
datasets = pkl.load(file)

trainloader, testloader, validloader = get_dataloaders(datasets, tran, batch_size = 30)


D_in, H, D_out = trainloader.dataset.shape[1] - 1, 30 , 2


model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)

lr = 0.005
loss_fn = nn.CrossEntropyLoss()
wd = 0.01
optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)


#trainloader, testloader, validloader = get_dataloaders(datasets, tran, batch_size = 30)
#dataloaders = trainloader, testloader, validloader
#ftr_accu, ftr_auc, ftr_cm, ftr_diff = trials(model, params, dataloaders, 1)
#print ("Accuracy after dropping feature", x)
#print ("Accuracy:", ftr_accu)
#print("AUC:", ftr_auc)
#print(ftr_cm)
#print("Overfitting", ftr_diff)
#print("Difference:", ftr_accu - best_all_accu)
#




for x in cols:
    comb_one = combined[[x, combined.columns[-1]]]  
    train, test, valid = get_partitions(comb_one, [0.6, 0.2, 0.2])
    datasets = train,test, valid
    trainloader, testloader, validloader = get_dataloaders(datasets, tran, batch_size = 30)
    dataloaders = trainloader, testloader, validloader
    ftr_accu, ftr_auc, ftr_cm, ftr_diff = trials(model, params, dataloaders, 1)
    print ("Accuracy after dropping feature", x)
    print ("Accuracy:", ftr_accu)
    print("AUC:", ftr_auc)
    print(ftr_cm)
    print("Difference:", ftr_accu - best_all_accu)
    ftr_imps[x] = (ftr_accu - best_all_accu, ftr_accu, ftr_auc, abs(ftr_diff) > 5)

file = open("feature_importances_individual", "wb")
pkl.dump(ftr_imps, file)

    
    
