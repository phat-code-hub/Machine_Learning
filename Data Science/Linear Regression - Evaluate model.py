from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
boston_dataset=load_boston()
boston=pd.DataFrame(boston_dataset.data,
                    columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target
X=boston
y=boston['MEDV']
X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
plt.style.use('ggplot')
plt.scatter(X_test,y_test,label='testing data',color='b')
# print(model.intercept_.round(2))
# print(model.coef_.round(2))
plt.xlabel('RM')
plt.ylabel('MEDV')
y_test_predict=model.predict(X_test)
plt.plot(X_test,y_test_predict,
         label='prediction',linewidth=3)
plt.legend(loc='upper left')
plt.show()
# Calculate Residual
residuals=y_test-y_test_predict
plt.scatter(X_test,residuals)
#plot horizontal line
plt.hlines(y=0,
           xmin=X_test.min(),
           xmax=X_test.max(),
           linestyle='--')
#set limit
plt.xlim((4,9))
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()
print(residuals[:5])
print(residuals.describe())
#Apply Means Square Error MSE 
print((residuals**2).mean().round(2))
# use MSE of sklearn
print(mean_squared_error(y_test,y_test_predict).round(2))
#Use R-square (score)
print(model.score(X_test,y_test).round(3))
print(model.score(X_test,y_test_predict))
#This score is calculated from below steps
mean_test_diff=((y_test-y_test.mean())**2).sum()
print(mean_test_diff)
res_sum=(residuals**2).sum()
print(res_sum)
print((1- (res_sum/mean_test_diff)).round(3)) # = score (X_test,y_test)
