import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')
from torchnet import meter
import pickle as pkl 



def preprocess(df):
    df['bare_nuclei'].replace({'?': np.nan}, inplace = True)
    df.dropna(inplace=True)
    df["bare_nuclei"] = df["bare_nuclei"].astype(int)
    df.drop(["id"], axis = 1, inplace=True)
    df["class"] = df["class"].map({2:0, 4:1})
    return df

dataset = pd.read_csv('data.csv')

dataset = preprocess(dataset)

file = open("datasets", "wb")

pkl.dump(dataset, file)

