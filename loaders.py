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
def get_dataloaders(datasets, transform = None, batch_size = 5, shuffle = True):
    """
    Takes a 3-tuple constaining Pandas Dataframes containing Training, Test and 
    Validation datasets respectively and returns a corresponding 3-tuple containing
    PyTorch DataLoader objects
    
    datasets: 3-tuple constaining Pandas Dataframes containing Training, Test and 
              Validation datasets respectively
    
    transform: List of PyTorch Transforms.Compose objects to be applied to
               the data
    
    Batch size: Batchsize to be used while training
    
    Shuffle: Shuffles the datasets if True
    
    Returns: Corresponding 3-tuple containingPyTorch DataLoader objects
    """
    
    train, test, valid = datasets
    trainset = WBCDataset(train, transform)
    testset = WBCDataset(test, transform)
    validset = WBCDataset(valid, transform)
    
    
    trainloader = DataLoader(trainset, shuffle= True, batch_size=batch_size, num_workers=4)
    testloader = DataLoader(testset, shuffle= True, batch_size=batch_size, num_workers=4)
    validloader = DataLoader(validset, shuffle= True, batch_size=batch_size, num_workers=4)
    
    return trainloader, testloader, validloader    

def get_partitions(df, partitions):
        """
        Partitions data into training, test and validation data in form of pandas
        dataframes
        
        partitions: A list containing three values between 0 and 1.0, where the
                    first, the second and the third element denotes the fraction 
                    of the data to be partitioned into Training, test and valid-
                    ation set respectively. The sum of the elements must equal 
                    1.0
        
        Returns: A 3-tuple containing pandas Dataframes containing the training, 
                 test and validation sets respectively
    
        """
        num_train = int(partitions[0]*df.shape[0])
        num_test = int(partitions[1]*df.shape[0])
        df_copy = df.copy()
        
        df_copy = df_copy.sample(frac = 1).reset_index(drop=True) #shuffle the data

        train_set = df[:num_train]
        test_set = df[num_train:num_train + num_test]
        valid_set = df[num_train + num_test:]
        
        return train_set, test_set, valid_set  
    
class loaders(object):
    """
    Loads the data from a CSV file. Capable of paritioned data either in form 
    of Pandas datframes, or PyTorch Dataloader object

    """
    def __init__(self, csv, preprocess = False):
        """
        Initialise the instance with a CSV file containing data, with an option
        to supply a pre-processing function for it
        
        csv:        CSV file containing the data
        preprocess: A function that takes a pandas dataframe as an input and 
                    outputs a pandas dataframe. Use it to pre-process your data
                    such as dataframe with pre-processed data is returned.
                    
        Returns: None
        """
        self.csv = csv
        self.df = pd.read_csv(csv)
        self.preprocess = preprocess
        if preprocess:
            self.df = self.preprocess(self.df)
    
    def get_partitions(self, partitions):
        """
        Partitions data into training, test and validation data in form of pandas
        dataframes
        
        partitions: A list containing three values between 0 and 1.0, where the
                    first, the second and the third element denotes the fraction 
                    of the data to be partitioned into Training, test and valid-
                    ation set respectively. The sum of the elements must equal 
                    1.0
        
        Returns: A 3-tuple containing pandas Dataframes containing the training, 
                 test and validation sets respectively
    
        """
        num_train = int(partitions[0]*self.df.shape[0])
        num_test = int(partitions[1]*self.df.shape[0])
        num_valid = int(partitions[2]*self.df.shape[0])
        df_copy = self.df.copy()
        
        df_copy = df_copy.sample(frac = 1).reset_index(drop=True) #shuffle the data

        train_set = self.df[:num_train]
        test_set = self.df[num_train:num_train + num_test]
        valid_set = self.df[num_train + num_test:]
        
        return train_set, test_set, valid_set
    
    def get_loaders(self, partitions, transform = None, batch_size = 5, shuffle = True):
        """
        Partitions data assosciated with the loader object into training, test 
        and validation data in form of PyTorch DataLoader objects. It can also 
        return DataLoader objects for a 3-tuple containing training, test and 
        validation data
        
        partitions: A list containing three values between 0 and 1.0, where the
                    first, the second and the third element denotes the fraction 
                    of the data to be partitioned into Training, test and valid-
                    ation set respectively. The sum of the elements must equal 
                    1.0
        
        
        transform: List of PyTorch Transforms.Compose objects to be applied to
                   the data
        
        batch_size: Batch size to be used while training
        
        shuffle:  Shuffle the dataset
                   
        
        Returns: A 3-tuple containing PyTorch DataLoader objects containing the
                 training, test and validation sets respectively
    
        """
        
        
        datasets = self.get_partitions(partitions)
        return get_dataloaders(datasets, transform = None, batch_size = None, shuffle = True)
        
    
def get_dloader(df, transform = None, batch_size = 30, shuffle = True):    
    dataset = WBCDataset(df, transform)
    dataloader = DataLoader(dataset, shuffle= True, batch_size=batch_size, num_workers=4)
    return dataloader
    