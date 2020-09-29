from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import  numpy as np
# import matplotlib.pyplot as plt
boston_dataset=load_boston()
boston=pd.DataFrame(boston_dataset.data,
                    columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target
X=boston[['RM']]
y=boston['MEDV']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
# 
model=LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_.round(2))
# print(X.shape)
# print(y.shape)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
print(model.coef_.round(2))
# #New Info and predict it
new_RM=np.array([6.5]).reshape(-1,1)
print(model.predict(new_RM))
# print(boston['RM'].describe())
# #test and predict for all homes
y_test_pre=model.predict(X_test)
print(y_test_pre.shape)
print(type(y_test_pre)) # == y_test.shape 1d numpy.array
print(y_test_pre[:10])