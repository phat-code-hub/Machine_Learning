# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
# from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
wine=pd.DataFrame(load_wine().data,columns=load_wine().feature_names)
print(wine.shape)
print(wine.columns)
print(wine.iloc[:,:3].describe())
#print(wine.loc[:,['alcohol','ash','magnesium']].describe())
print(wine.iloc[:,:3].info())
X2=wine.iloc[:,[0,5]]
#print(X2.head())
X1=wine.loc[:,['alcohol','total_phenols']]
#print(X1.head())
X=wine[['alcohol','total_phenols']]
#print(X.head())
scale=StandardScaler()
#compute the mean and std to be used later for scaling
scale.fit(X)
print('Phat',scale.scale_)
print(scale.mean_)
print(scale.scale_)
X_scaled=scale.transform(X)
print(X_scaled[:,:])
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
plt.scatter(X['alcohol'],X['total_phenols'])
plt.scatter(X_scaled[:,0],X_scaled[:,1],color='r')
plt.show()