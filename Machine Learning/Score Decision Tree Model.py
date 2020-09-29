# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score,recall_score,f1_score


df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y=df['Survived'].values

# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=22)
kf=KFold(n_splits=5,shuffle=True,random_state=10)
dt_accuracy_scores=[]
dt_precision_scores=[]
dt_recall_scores=[]
lr_accuracy_scores=[]
lr_precision_scores=[]
lr_recall_scores=[]

for train_index, test_index in kf.split(X):
    X_train, X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    #Decision Tree
    dt=DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    dt_y_pred=dt.predict(X_test)
    dt_accuracy_scores.append(dt.score(X_test,y_test))
    dt_precision_scores.append(precision_score(y_test,dt_y_pred))
    dt_recall_scores.append(recall_score(y_test,dt_y_pred))
    #LogisticsRegression
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    lr_y_pred=lr.predict(X_test)
    lr_accuracy_scores.append(lr.score(X_test,y_test))
    lr_precision_scores.append(precision_score(y_test,dt_y_pred))
    lr_recall_scores.append(recall_score(y_test,dt_y_pred))
print('Decision Tree:')
print('\taccuracy: ',np.mean(dt_accuracy_scores))
print('\tprecision: ',np.mean(dt_precision_scores))
print('\trecall: ',np.mean(dt_recall_scores))
print('Logistics Regrression:')
print('\taccuracy: ',np.mean(lr_accuracy_scores))
print('\tprecision: ',np.mean(lr_precision_scores))
print('\trecall: ',np.mean(lr_recall_scores))