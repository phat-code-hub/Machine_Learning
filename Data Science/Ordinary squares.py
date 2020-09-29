import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
n,p=[int(x) for x in input().split()]
X=[]
for i in range(n):
    X.append([float(x) for x in input().split()])
y=[float(x)for x in input().split()]

model=LinearRegression()
model.fit(X,y)
print(model.coef_)
print(model.intercept_)