#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 00:29:32 2017

@author: ayooshmac
"""

import xgboost
import pickle as pkl
from tabulate import tabulate

model = xgboost.XGBClassifier(learning_rate = 0.05, n_estimators=50)
datasets = pkl.load(open("../data/datasets", "rb"))

train, test, valid = datasets

train_X = train.drop("class", axis= 1)
train_Y = train["class"]

test_X = test.drop("class", axis = 1)
test_Y = test["class"]

model.fit(train_X, train_Y)

ftr_imps = model.feature_importances_


ftr_imps = list(zip(train.columns, ftr_imps))

ftr_imps = sorted(ftr_imps, key = lambda x: x[1], reverse = True)

file = open("feature_importance_gbm.txt", "w")

li = [("Feature", "GBM Importance")] + li
 
a = tabulate(li, headers="firstrow")
 
print(a, file = file)
file.close()