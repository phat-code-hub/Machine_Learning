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
y_pred=model.predict(X)
#Create boolean array of whether model predict each passenger correctly or not
# to calculate percent
correctly = y==y_pred
print('Corect cases: ',correctly.sum())
print('Correct percent: ', correctly.sum()/y.shape[0])
#Can get same result by use Score
print('Score of model: ',model.score(X,y))
