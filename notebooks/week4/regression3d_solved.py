#%%
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
#%%
df = pd.read_csv('/Users/enaj/SUND/CompNeuro course/CompNeuro22/notebooks/week4/Stars.csv')
df.head()

target_feature = 'A_M'
feature_cols = ['Temperature', 'Size']
Y = df[target_feature].values
X = df[feature_cols].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = MLPRegressor(max_iter=1000, # Maximum number of steps we update our model
                    activation="tanh", # activation function
                    early_stopping=False, # Should the training stop if loss converges? 
                    hidden_layer_sizes=(500,500,500), # Hidden layers size
                    learning_rate_init=0.00005, # learning rate
                    learning_rate = 'adaptive',
                    )


model.fit(X_train, y_train)

print('Score on training set: ', model.score(X_train, y_train))
print('Score on test set: ', model.score(X_test, y_test))

#%%
input_x = np.array([np.linspace(X[:,0].min(), X[:,0].max(), 10000), np.linspace(X[:,1].min(), X[:,1].max(), 10000)]).T
pred_y = model.predict(input_x)

#%%
# df.plot.scatter(feature_cols[0],target_feature, figsize=(10,7), label='Data', color='black')
# plt.plot(input_x[:,0],pred_y, label='Model prediction', color='red')

# df.plot.scatter(feature_cols[1],target_feature, figsize=(10,7), label='Data', color='black')
# plt.plot(input_x[:,1],pred_y, label='Model prediction', color='red')

#%%
fig = pyplot.figure( figsize=(10,7))
ax = Axes3D(fig)
ax.scatter(df[feature_cols[0]].values, df[feature_cols[1]].values, df[target_feature].values, color='xkcd:black', s=50, alpha=1, marker='*' )
ax.plot(input_x[:,0], input_x[:,1], pred_y, color='black', alpha=0.9 )
ax.set_xlabel("Temperature")
ax.set_ylabel("Size")
ax.set_zlabel("A_M")
ax.legend(markerscale=6., prop={"size":12})
pyplot.show()

#%%

# pred_y.shape