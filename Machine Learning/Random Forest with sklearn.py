# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer_data=load_breast_cancer()
df=pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
df['target']=cancer_data['target']
X=df[cancer_data.feature_names].values
y=cancer_data['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=101)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

first_row=X_test[0]
print('Prediction: ', rf.predict([first_row]))
print('true value:' , y_test[0])
print('Accuracy score: ',rf.score(X_test,y_test).round(4))

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
print('Prediction: ', dt.predict([first_row]))
print('true value:' , y_test[0])
print('Accuracy score: ',dt.score(X_test,y_test).round(4))
