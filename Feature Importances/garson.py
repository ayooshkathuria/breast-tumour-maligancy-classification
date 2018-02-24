#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 00:39:33 2017

@author: Ayoosh
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
from loaders import *


import torch 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torchvision import transforms

fle  = open("model_params", "rb")
model_params = pkl.load(fle)

params_li = list(model_params.parameters())

num_hidden = params_li[0].data.shape[0]

num_ftrs = params_li[0].data.shape[1]

d = torch.zeros(num_ftrs,)

for unit in range((num_hidden)):
    a = params_li[0][unit].data
    b = params_li[2][1][unit].data
    c = a*b
    c = c/c.abs().sum()
    d += c

print (d/d.abs().sum())
    