# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.head())
# make  columns as numerical
df['male']=df['Sex']=='male'
# next , take all features and create np.array called X
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
# take the target Y:
Y=df['Survived'].values
print(X[:5])
print(Y[:5])