# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
feature_names=['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']
X=df[feature_names]
y=df['Survived']

param_grid= {
        'max_depth':[5,15,25],
        'min_samples_leaf':[1,3],
        'max_leaf_nodes':[10,20,35,50]
    }
dt=DecisionTreeClassifier()
gs=GridSearchCV(dt,param_grid,scoring='f1',cv=5)
gs.fit(X,y)
print("best Params: ",gs.best_params_)
print("best score: ",gs.best_score_.round(4))