import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


import pickle as pkl
a = open("datasets", "rb")
a = pkl.load(a)
comb_values = pd.concat(a)

features = comb_values.iloc[:,:-1]
labels = comb_values["class"]

pca = PCA(n_components=4)
tsne = TSNE(perplexity=30)
transformed_features = pca.fit_transform(features.values)
transformed_features = pd.DataFrame(transformed_features)
transformed_features = tsne.fit_transform(transformed_features.values)
tf = pd.DataFrame(transformed_features, columns = ["tsne1", "tsne2"])
color = {1: "red", 0: "blue"}
plt.scatter(tf["tsne1"], tf["tsne2"], color= labels.apply(lambda x: color[x]))


