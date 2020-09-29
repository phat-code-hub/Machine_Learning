# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']]
y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=22)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
XX=[[3,True,22,1,0,7.25]]
print(model.predict(XX))