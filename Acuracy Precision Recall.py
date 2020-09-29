# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
print(df.head())
#Dataset
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
# take the target Y:
y=df['Survived'].values

model=LogisticRegression()
model.fit(X,y)
y_pred=model.predict(X)
print('Accuracy: ',accuracy_score(y,y_pred))
print('Precision : ',precision_score(y,y_pred))
print('Recall: ',recall_score(y,y_pred))
print('F1 score: ',f1_score(y,y_pred))
print()
print('Confusion matrix:')
#2D matrix
# [FN FP]
# [TN TP]
print(confusion_matrix(y,y_pred))
qr=df[(df['Sex']=='female') & (df['Age']>30)]
print(qr['Survived'].describe())
abc=qr.groupby('Survived')
print(abc['Sex'].count)