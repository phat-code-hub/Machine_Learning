#This part find coefficent a,b, c of Linear Equation
import pandas as pd
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.head())
#Create Feature and Target and  X,Y
X=df[['Fare','Age']].values
y=df['Survived'].values
#Initial class and fit it
model=LogisticRegression()
model.fit(X,y)
#find [a,b],[c] of linear Equation ax+by+c=0
print(model.coef_,model.intercept_)