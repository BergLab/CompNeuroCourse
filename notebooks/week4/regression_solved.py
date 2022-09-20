#%%
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

#%%
# data = np.loadtxt('/Users/enaj/SUND/CompNeuro course/CompNeuroBook/notebooks/week4/Stars.csv', skiprows=1)
df = pd.read_csv('Stars.csv')
df.head()
#%%
# Generate data
X = np.linspace(0,50,10000).reshape(-1,1)
Y = np.sin(X)

# Define neural network model
model = MLPRegressor(max_iter=1000, activation="tanh",
                    early_stopping=True, hidden_layer_sizes=(100,100),
                    )

# Train model
model.fit(X, Y.ravel())

# Print Score
print('Score on training: ', model.score(X, Y))

# Predict data values with model and plot along original data
pyplot.figure(figsize=(10,8))
pyplot.scatter(X,Y, label='Original data')
input_x = np.linspace(X.min(), X.max()*2, 10000).reshape(-1,1)
pred_y = model.predict(input_x)
pyplot.plot(input_x,pred_y, label='Model prediction', color='red')
pyplot.legend(fontsize=16)

#%%



#%%
target_feature = 'A_M'
feature_cols = 'Luminosity'
X = df[feature_cols].values.reshape(-1,1)
Y = df[target_feature].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

model = MLPRegressor(max_iter=200, # Maximum number of steps we update our model
                    activation="tanh", # activation function
                    early_stopping=False, # Should the training stop if loss converges? 
                    hidden_layer_sizes=(300,300,300), # Hidden layers size
                    learning_rate_init=0.00005, # learning rate
                    learning_rate = 'adaptive',
                    )


model.fit(X_train, y_train)

print('Score on training set: ', model.score(X_train, y_train))
print('Score on test set: ', model.score(X_test, y_test))

df.plot.scatter(feature_cols,target_feature, figsize=(10,7), label='Data', color='black')
input_x = np.linspace(0, X.max(), 100000).reshape(-1,1)
pred_y = model.predict(input_x)
plt.plot(input_x,pred_y, label='Model prediction', color='red')

##########
##########
##########
#%%
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

df.plot.scatter(feature_cols[0],target_feature, figsize=(10,7), label='Data', color='black')
plt.plot(input_x[:,0],pred_y, label='Model prediction', color='red')

df.plot.scatter(feature_cols[1],target_feature, figsize=(10,7), label='Data', color='black')
plt.plot(input_x[:,1],pred_y, label='Model prediction', color='red')
#%%

#%%
target_feature = 'A_M'
feature_cols = ['Size']
Y = df[target_feature].values
X = df[feature_cols].values.reshape(-1,1)
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
df.plot.scatter(feature_cols,target_feature, figsize=(10,7), label='Data', color='black')
input_x = np.linspace(0, X.max(), 10000).reshape(-1,1)
pred_y = model.predict(input_x)
plt.plot(input_x,pred_y, label='Model prediction', color='red')
plt.legend(fontsize=12)






#%%
target_feature = 'A_M'
feature_cols = ['Size']
Y = df[target_feature].values
X = df[feature_cols].values.reshape(-1,1)
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
df.plot.scatter(feature_cols,target_feature, figsize=(10,7), label='Data', color='black')
input_x = np.linspace(0, X.max(), 10000).reshape(-1,1)
pred_y = model.predict(input_x)
plt.plot(input_x,pred_y, label='Model prediction', color='red')
plt.legend(fontsize=12)
