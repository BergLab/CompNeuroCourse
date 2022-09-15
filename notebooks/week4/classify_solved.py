#%%
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from matplotlib import cm
import plotly.express as px
import seaborn as sns

from sklearn.inspection import DecisionBoundaryDisplay

#%%
# data = np.loadtxt('/Users/enaj/SUND/CompNeuro course/CompNeuro22/notebooks/week4/Stars.csv', skiprows=1)
df = pd.read_csv('Stars.csv')
df.head()

#%%
# data = np.loadtxt('AlephBtag_MC_small_v2.csv', skiprows=1)
# X = data[:, :-1]
# y = data[:, -1]
feature_cols = ['Temperature', 'Size', 'Luminosity', 'A_M']
X = df[feature_cols]
Y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# %%
model = MLPClassifier(solver='adam', 
                    hidden_layer_sizes=(30,30), 
                    alpha=1e-3,
                    max_iter=100000,
                    early_stopping=True)

model.fit(X_train, y_train)

print('Accuracy on training: ', model.score(X_train, y_train))
print('Accuracy on test: ', model.score(X_test, y_test))

# %%
# %%
feature_cols = ['Luminosity', 'A_M']
X = df.loc[:, feature_cols]
X.shape
X.head()
# %%
feature_cols = ['Luminosity', 'A_M']
X = df[feature_cols]
X.shape
X.head()
#%%
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X, Y, cv=20, return_times=True)

# pyplot.plot(train_sizes,np.mean(train_scores,axis=1))
pyplot.plot(fit_times, test_scores)

#%%

#%%
def NeuralClass(X,y):
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2)
    mlp=MLPClassifier(
        activation="relu",
        max_iter=3000, 
        validation_fraction=0.2, 
        early_stopping=True)
    mlp.fit(X_train,y_train)
    print (mlp.score(X_train,y_train))
    pyplot.plot(mlp.loss_curve_, label='Loss')
    pyplot.plot(mlp.validation_scores_, label='Validation')

NeuralClass(X,Y)

#%%
df = pd.read_csv('Stars.csv')
Y = df['Spectral_Class'].to_numpy()
# Y = df['Type'].to_numpy()
feature_cols = ['Temperature', 'Size', 'Luminosity',]
X = df[feature_cols].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#%%
model = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30,30) , alpha=1e-2, 
                    max_iter=100000, early_stopping=True)
model.fit(X_train, y_train)

print('Accuracy on training: ', model.score(X_train, y_train))
print('Accuracy on test: ', model.score(X_test, y_test))

#%%
print(len(model.loss_curve_))
print(len(model.validation_scores_))
pyplot.plot(model.loss_curve_)
#%%
model.best_validation_score_


#%%
# random_seed = 12
# random_seed = 452
random_seed = np.random.randint(10000)
print(random_seed)
feature_cols = ['Temperature', 'A_M']
target_feature = 'Spectral_Class'
X = df[feature_cols].values
Y = df[target_feature].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

model = MLPClassifier(solver='adam', 
                    # hidden_layer_sizes=(100,100,100), 
                    hidden_layer_sizes=(500,500,500), 
                    activation='tanh',
                    max_iter=10000,
                    learning_rate = 'adaptive',
                    learning_rate_init=0.00001,
                    # learning_rate_init=0.00005,
                    # learning_rate_init=0.0001,    
                    early_stopping=False,
                    random_state = random_seed)
model.fit(X_train, y_train)

print('Accuracy on training: ', model.score(X_train, y_train))
print('Accuracy on test: ', model.score(X_test, y_test))


plt.figure(figsize=(12,8))
DecisionBoundaryDisplay.from_estimator(
    model,
    X,
    cmap = 'Set3',
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    xlabel=feature_cols[0],
    ylabel=feature_cols[1],
    eps=0.5,
)

colors = plt.cm.get_cmap('hsv', len(df[target_feature].unique()))
for index, startype in enumerate(df[target_feature].unique()):
    stars_one_type = df[df[target_feature] == startype][feature_cols].values
    plt.scatter(stars_one_type[:, 0], stars_one_type[:, 1], color = colors(index), edgecolors="k", label=startype)

plt.legend(fontsize=12)
plt.show()




#%%
params = {'criterion': 'entropy', 'max_depth': 15, 'max_features': 2, 'splitter': 'best', 'random_state': 42}
classifier =  Classification(predictor = 'dt',params=params)
classifier.fit(features, labels)
result = classifier.result()
accuracy_scores[result['Classifier']] = result['Accuracy']

#%%
features_id = ['Size', 'A_M', 'Temperature']
features = df[features_id]
labels = df['Type']
params = {'criterion': 'entropy', 'max_depth': 15, 'max_features': 2, 'splitter': 'best', 'random_state': 42}
classifier =  Classification(predictor = 'dt',params=params)
classifier.fit(features, labels)
result = classifier.result()
accuracy_scores[result['Classifier']] = result['Accuracy']

#%%

plt.cm.get_cmap('hsv', 5)
