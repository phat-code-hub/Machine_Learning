# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
X,y=make_classification(n_features=2,n_redundant=0,n_informative=2,
                        random_state=3)
X1=X[y==0]
X2=X[y==1]
plt.scatter(X[y==0][:,0],X[y==0][:,1],s=100,edgecolor='k')
plt.scatter(X[y==1][:,0],X[y==1][:,1],s=100,edgecolor='k',marker='^')
plt.show()