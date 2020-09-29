# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
# print(df.head())
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']] \
    .values
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=7)
#X_train,X_test,y_train,y_test=train_test_split(X,y)
model1=LogisticRegression()
model1.fit(X_train,y_train)
y_pred_proba1=model1.predict_proba(X_test)
print('Model 1 AUC score: ',roc_auc_score(y_test,y_pred_proba1[:,1]).round(3))
#-----------------------------------
#fir 2 features Pclass and male only
model2=LogisticRegression()
model2.fit(X_train[:,0:2],y_train)
y_pred_proba2=model2.predict_proba(X_test[:,0:2])
print('Model 2 AUC score: ',roc_auc_score(y_test,y_pred_proba2[:,1]).round(3))
#-------------------------------------
#Plot result
plt.figure(figsize=(6,5))
fpr1,tpr1,threshole1=roc_curve(y_test,y_pred_proba1[:,1])
fpr2,tpr2,threshole2=roc_curve(y_test,y_pred_proba2[:,1])
plt.plot(fpr1,tpr1,c='r')
plt.plot(fpr2,tpr2,c='Orange')
plt.plot([0,1],[0,1],linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('1-Specitivity')
plt.ylabel('Sensitivity')
plt.plot()