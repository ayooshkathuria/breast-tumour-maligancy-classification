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
    
    def get_loss(self, dataloader):
        loss = 0
        for data in dataloader:
            x,y = data
            x,y = Variable(x), Variable(y)
            outputs = self.model(x) 
            loss += self.loss_fn(outputs, y).data[0]
        return loss/float(dataloader.dataset.shape[0])
    
    def metrics(self, testloader, accuracy = True, auc = False, conf_matrix = False):
        am = meter.AUCMeter()
        cm = meter.ConfusionMeter(2)
        correct = 0
        total = 0
        for data in testloader:   
           
            x,y = data
        
            y_ = self.model(Variable(x))
            _, predicted = torch.max(y_.data, 1)
          
            cm.add(y_.data, y)
            
            am.add(y_.data[:,1].clone(),y)
            total += y.size(0)
            correct += (predicted == y).sum()
        print (correct, total)
        if accuracy:
            print("Accuracy for the model is", round(correct/float(total)*100, 4), correct, "/", total)
        
        if auc:
            print("Area under ROC curve for the given model is", round(am.value()[0],4))
        
        if conf_matrix:
            print ("Confusion Matrix for the given model is\n", cm.value())

    def metrics_val(self, testloader):
            am = meter.AUCMeter()
            cm = meter.ConfusionMeter(2)
            correct = 0
            total = 0
            for data in testloader:   
                x,y = data
                y_ = self.model(Variable(x))
                _, predicted = torch.max(y_.data, 1)
                cm.add(y_.data, y)
                am.add(y_.data[:,1].clone(),y)
                total += y.size(0)
                correct += (predicted == y).sum()
            
            cor_tot = str(correct) + "/" + str(total)
        
            return round(correct/float(total)*100, 4), cor_tot, round(am.value()[0],4), cm.value()
        
    def get_logs(self):
        return self.losses, \
            self.losses_test, \
            self.accus, \
            self.accus_train
    
    def plot(self, logs):
        losses, losses_test, accus, accus_train = logs
        steps = range(1, len(losses)+1)
        
        plt.plot(steps, losses, color = 'r', label = 'Training Loss')
        plt.plot(steps, losses_test, color = 'b', label = 'Test Loss')
        plt.xlabel("Epochs")
        plt.legend(['Training Loss', 'Test Loss'])
#            
        plt.figure(2)
        plt.xlabel("Epochs")
        plt.plot(steps, accus, color = 'b', label = 'Train Accuracy')
        plt.plot(steps, accus_train, color = 'r', label = 'trTrain Accuracy')
        plt.legend(['Test accuracy', 'Training Accuracy'])
        
    def decision_boundary_2d(self, df, f1, f2, label = "class", h = 0.2, plot = True):
        """
        Renders a 2-dimensional decision boundary generated by
        the Neural Network for given data.
        
        df: Dataframe containing the data with labels as well as 
            the class. 
        
        xx: Column name of the feature to be plotted on the x-axis
        yy: Column name of the label to be plotted on the y-axis
        label: name of the column containing the class
        
        Returns: Plots the decision boundary with the points colored
                 with class
        """
        
        color = {1: "red", 0: "blue"}
        
        x = df[f1]
        y = df[f2]
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
        
        grid = np.c_[xx.ravel(), yy.ravel()] 
        grid_tensor = Variable(torch.Tensor(grid))
        
        results = torch.max(self.model(grid_tensor).data, 1)[1].numpy()
        if plot:
            plt.contourf(xx, yy, results.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.8)
            plt.scatter(df[f1], df[f2], c=df[label].apply(lambda x: color[x]))
        
        else:
            return results