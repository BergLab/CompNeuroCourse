#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

#%%

################
# 2D PCA #######
################

df = pd.read_csv('/Users/enaj/SUND/CompNeuro course/CompNeuroBook/notebooks/week4/Stars.csv')
df.head()

feature_cols = ['Temperature', 'Size', 'A_M']
data = df[feature_cols].values
labels = df['Type'].values

l = []
for label in labels:
    if label == 'Main Sequence':
        l.append(0)
    elif label == 'Brown Dwarf':
        l.append(1)
    elif label == 'Super Giants':
        l.append(2)
    elif label == 'White Dwarf':
        l.append(3)
    elif label == 'Red Dwarf':
        l.append(4)
    elif label == 'Hyper Giants':
        l.append(5)
    else:
        raise AssertionError


pca = PCA(n_components=3)
pca.fit(data)

# print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

x_pca_ = pca.transform(data)

plt.figure(figsize=(8,6))
plt.scatter(x_pca_[:,0], x_pca_[:,1],c=l, cmap='rainbow', s=10)
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

#%%

################
# 3D PCA #######
################

    
df = pd.read_csv('/Users/enaj/SUND/CompNeuro course/CompNeuroBook/notebooks/week4/Stars.csv')
df.head()

feature_cols = ['Temperature', 'Size', 'A_M']
data = df[feature_cols].values
labels = df['Type'].values
labels_unique = df['Type'].unique()


l = []
for label in labels:
    if label == 'Main Sequence':
        l.append(0)
    elif label == 'Brown Dwarf':
        l.append(1)
    elif label == 'Super Giants':
        l.append(2)
    elif label == 'White Dwarf':
        l.append(3)
    elif label == 'Red Dwarf':
        l.append(4)
    elif label == 'Hyper Giants':
        l.append(5)
    else:
        raise AssertionError



pca = PCA(n_components=3)
pca.fit(data)

# print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

x_pca_ = pca.transform(data)

fig = plt.figure(figsize=(8,12))
ax = Axes3D(fig)

scatter = ax.scatter(x_pca_[:,0], x_pca_[:,1], x_pca_[:,2], c=l)

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
ax.legend(markerscale=6., prop={"size":12})

# plt.legend(labels_unique)

plt.show()
