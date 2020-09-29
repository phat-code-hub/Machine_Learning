# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
# print(df.head())
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']] \
    .values
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=7)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred_proba=model.predict_proba(X_test)
fpr,tpr,threshole=roc_curve(y_test,y_pred_proba[:,1])
plt.plot(fpr,tpr,c='r')
plt.plot([0,1],[0,1],linestyle='--')
plt.scatter([fpr[17],fpr[42],fpr[59]],[tpr[17],tpr[42],tpr[59]])
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('1-Specitivity')
plt.ylabel('Sensitivity')
plt.plot()