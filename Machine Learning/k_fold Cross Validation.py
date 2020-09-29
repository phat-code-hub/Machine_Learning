# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd

df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
X=df[['Age','Fare']].values[:6]
y=df['Survived'].values[:6]
kf=KFold(n_splits=3,shuffle=True)
# for train,test in kf.split(X):
#     print(train,test)
splits=list(kf.split(X))
first_split=splits[0]
train_indices,test_indices=first_split
print('Training set indices: ',train_indices)
print('Testing set indices: ',test_indices)

X_train=X[train_indices]
X_test=X[test_indices]
y_train=y[train_indices]
y_test=y[test_indices]

# print('X train: ')
# print(X_train)
# print('X test: ',X_test)
# print('y train: ')
# print(y_train)

.# print('y test: ',y_test)

model=LogisticRegression()
model.fit(X_train,y_train)
print('Score: ', model.score(X_test,y_test))
