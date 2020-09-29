# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer_data=load_breast_cancer()
df=pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
df['target']=cancer_data['target']
X=df[cancer_data.feature_names].values #shape (569,30)
df['target']=cancer_data['target']
y=df['target'].values
new_X=X[:10]
# model=LogisticRegression() # bao error do khong co Converge
#add solver to fix error
model=LogisticRegression(solver="liblinear") # good for small data , 
#big one is saga, newton-cg
model.fit(X,y)
#Model.predict([X[0]]) # predict first row
#print('Prediction for datapoint 0: ',model.predict([X[:5]]))
print('Prediction for datapoint 0: ',model.predict(new_X))
print("Score :", model.score(X,y).round(3))
#print(X[0])
#print(df.iloc[0,:])
 