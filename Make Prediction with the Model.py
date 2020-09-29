# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.head())
#Take features to X
df['male']=df['Sex']=='male'
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
# take the target Y:
y=df['Survived'].values
#take model
model=LogisticRegression()
model.fit(X, y)
#predict all
#print(model.predict(X))
#predict specific target
print(model.predict([[3,True,22.0,1,0,7.25]])) # 0  not survived
print(model.predict([[1,False,38.0,1,0,71.28]])) # 1 survived
print(model.predict(X[:5]))
#compare with the target 
print(y[:5])