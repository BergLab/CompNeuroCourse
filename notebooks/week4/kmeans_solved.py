#%%
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from matplotlib import cm
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


#%%
df = pd.read_csv('/Users/enaj/SUND/CompNeuro course/CompNeuro22/notebooks/week4/Stars.csv')
df.head()

#%%
# features = ['A_M', 'Temperature']
features = ['A_M', 'Size', 'Temperature']
data = df[features].values

nb_clusters = 4
Kmean = KMeans(n_clusters=nb_clusters)
Kmean.fit(data)

#%%
clusters_centers = Kmean.cluster_centers_ # aka the centroid
clusters_centers.shape
clusters_centers
#%%

fig = plt.figure(figsize=(8,12))
ax = Axes3D(fig)

ax.scatter(data[:,0], data[:,1], data[:,2])
colors = ['red', 'magenta', 'black', 'green']
for i in range(nb_clusters):
    ax.scatter(clusters_centers[i,0], clusters_centers[i,1],clusters_centers[i,2], s=200, c=colors[i], marker='s')


ax.set_xlabel("A_M")
ax.set_ylabel("Size")
ax.set_zlabel("Temperature")
ax.legend(markerscale=6., prop={"size":12})
plt.show()
#%%
data.shape