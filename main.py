import pickle as pkl 
from custom_model import *
from loaders import *
import time


import torch 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torchvision import transforms
from sklearn.decomposition import PCA 
from torchvision import transforms 

tran = transforms.Compose([transforms.ToTensor()])


a = open("data/datasets", "rb")
datasets = pkl.load(a)

drop_cols = ["marg_adhesion", "single_epith_cell_size", "mitoses"]

         
for x in datasets:
    x.drop(drop_cols, axis = 1, inplace = True)


datasets = [pca_dataframe(x,2).iloc[:,:] for x in datasets]


tran = transforms.Compose([transforms.ToTensor()])


trainloader, testloader, validloader = get_dataloaders(datasets, tran, batch_size = 30)


comb_data = pd.concat(datasets)
combset = WBCDataset(comb_data, tran)
combloader = DataLoader(combset, shuffle= True, batch_size=30, num_workers=4)

D_in, H, D_out = trainloader.dataset.shape[1] - 1, 30, 2
 
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)
lr = 0.1
loss_fn = nn.CrossEntropyLoss()
wd = 0.01
optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 2/float(12))
        m.bias.data.normal_(0, 2/float(12))
        
a = custom_model(model, loss_fn)
a.model.apply(init_weights)
start=  time.time()

a.train(trainloader, testloader, validloader, optimizer, 30, plot = True)
finish = time.time()

accuracy, ct, auc, cm = a.metrics_val(testloader)

print ("Train Accuracy", a.metrics_val(trainloader)[0], a.metrics_val(trainloader)[1])
print ("Test Accuracy: ", accuracy, ct)
print ("Validation Accuracy", a.metrics_val(validloader)[0], a.metrics_val(validloader)[1])
print ("Combined Accuracy", a.metrics_val(combloader)[0], a.metrics_val(combloader)[1])
print ("AUROC:", auc, "\nConfusion Matrix\n", cm)
print ("Time taken to train the network", finish - start)


#model = pkl.load(open("model-2-dim", "rb"))
model = a.model
a.plot(a.get_logs())

plt.figure(3)

color = {1: "red", 0: "blue"}

a.decision_boundary_2d(comb_data, "PCA0", "PCA1")