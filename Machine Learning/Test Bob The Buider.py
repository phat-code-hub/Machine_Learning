import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

n=int(input())
X=[]
for i in range(n):
    X.append([float(x) for x in input().split()])
y=[int(x) for x in input().split()]
datapoint=[float(x) for x in input().split()]
#------------------------------------------------
X_data=pd.DataFrame(X)
y_data=np.array(y)
y_test=np.array(datapoint).reshape(1,-1)
model=LogisticRegression(solver="liblinear")
model.fit(X_data,y_data)
y_pred=[str(x) for x in model.predict(y_test)]
print(' '.join(y_pred))