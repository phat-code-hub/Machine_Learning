# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#---------------------------------------------------------

df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
# print(df.head())
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']] \
    .values
y=df['Survived'].values
model=LogisticRegression()
model.fit(X,y)
y_pred=model.predict(X)
print('Accuracy :',accuracy_score(y,y_pred).round(4))
print('Precision: ',precision_score(y,y_pred).round(4))
print('Recall: ',recall_score(y,y_pred).round(4))
print('F1: ' ,f1_score(y,y_pred).round(4))
#--------------------------------------------------
print('Report:')
print(classification_report(y,y_pred))
#--------------------------------------------------
print('Confusion Matrix:')
print(confusion_matrix(y,y_pred))
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=7)
model1=LogisticRegression()
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
print('Accuracy 1:',accuracy_score(y_test,y_pred1).round(4))
print('Precision 1: ',precision_score(y_test,y_pred1).round(4))
print('Recall 1: ',recall_score(y_test,y_pred1).round(4))
print('F1: ' ,f1_score(y_test,y_pred1).round(4))
#--------------------------------------------------------
# Sensitivity and Specificity
sensitivity_score=recall_score
print('Sensivity score: ',sensitivity_score(y_test,y_pred1).round(4))
# the 2nd array is recall
#Create function
def specificity_score(Xtrue,ypred):
    p,r,f,s=precision_recall_fscore_support(Xtrue,ypred)
    return r[0].round(4)
print('Specificity: ',specificity_score(y_test,y_pred1))
#Adjusting Logistics RegressionThreshold using predict_proba
model2=LogisticRegression()
model2.fit(X_train,y_train)
print('Predict proba: ')
y_pred2=model2.predict_proba(X_test)[:,1]>0.75
print(X_test[:10])
print(y_pred2[:10])
print('precision proba: ',precision_score(y_test,y_pred2).round(4))
print('recall proba: ',recall_score(y_test,y_pred2).round(4))





