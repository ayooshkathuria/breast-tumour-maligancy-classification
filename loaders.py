#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:45:23 2017

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
tran = transforms.Compose([transforms.ToTensor()])
from sklearn.decomposition import PCA 


class WBCDataset(Dataset):
    """
    Loads the Wisconsin Breast Cancer dataset from a csv file into a PyTorch Dataset 
    Object which can then be passed to an iterable DataLoader object.
    """
    def __init__(self, df, transform = None):
        """
        Initialises the WBCDataset Instance from 
        
        df: dataframe containing the data
        Transform: default None, PyTorch transform object to be apply on the data
        
        """
        
        self.df = df
        self.shape = df.shape
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        x = self.df.drop(["class"], axis = 1)
        y = self.df["class"]
        x = (torch.from_numpy(x.iloc[idx].astype(np.float32).values))
        y = int(y.iloc[idx])
   
        sample = x,y
        
        return sample