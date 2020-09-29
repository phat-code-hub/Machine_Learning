# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score,recall_score,accuracy_score


df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y=df['Survived'].values

kf=KFold(n_splits=5,shuffle=True,random_state=10)

for criterion in ['gini','entropy']:
    print('Decision Tree - {}'.format(criterion))
    dt_accuracy_scores=[]
    dt_precision_scores=[]
    dt_recall_scores=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        #Decision Tree
        dt=DecisionTreeClassifier(criterion=criterion)
        dt.fit(X_train,y_train)
        dt_y_pred=dt.predict(X_test)
        dt_accuracy_scores.append(accuracy_score(y_test,dt_y_pred))
        dt_precision_scores.append(precision_score(y_test,dt_y_pred))
        dt_recall_scores.append(recall_score(y_test,dt_y_pred))
    print('\taccuracy: ',np.mean(dt_accuracy_scores))
    print('\tprecision: ',np.mean(dt_precision_scores))
    print('\trecall: ',np.mean(dt_recall_scores))
    print()