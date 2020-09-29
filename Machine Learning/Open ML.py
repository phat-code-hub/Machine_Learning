# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
# import numpy as np

X,y=fetch_openml('mnist_784',version=1,return_X_y=(True))
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2)

# print(X.shape,y.shape)
# print(np.min(X),np.max(X))
# print(y[0:5])

#Limit data 
# X5=X[y<='3']
# y=y[y<='3']

mlp=MLPClassifier(
    hidden_layer_sizes=(6,),
    max_iter=200,alpha=1e-4,
    solver='sgd',random_state=2)
mlp.fit(X_train,y_train)
# print(mlp.coefs_)
# print(len(mlp.coefs_))
# print(mlp.coefs_[0].shape)

fig,axes = plt.subplots(2,3,figsize=(5,4))
for i,ax in enumerate(axes.ravel()):
    coef=mlp.coefs_[0][:,i]
    ax.matshow(coef.reshape(28,28),cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i+1)
plt.show()