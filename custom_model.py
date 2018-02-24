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
    

