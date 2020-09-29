# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
cancer_data=load_breast_cancer()
print(cancer_data.keys())
#show info
# print(cancer_data['filename'])
#print(cancer_data['DESCR'])
print(cancer_data['data'].shape) # 569,30
print(cancer_data['feature_names'])
#Create Pandas.DataFrames from these feature data
df=pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
#add new columns target from target feature of data
df['target']=cancer_data['target']
print(df.head())
#Build Logistic Regression Model
X=df[cancer_data.feature_names].values
y=df['target'].values
# Fit data
model=LogisticRegression(solver='liblinear')
model.fit(X,y)
#Predict data, note that predict take 2d array
print('prediction for datapoint 0: ',model.predict([X[0]]))
print('Score: ', model.score(X,y))