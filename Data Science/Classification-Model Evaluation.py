import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
iris=pd.read_csv('../iris.csv')
X=iris[['PetalLength','PetalWidth']]
y=iris['Species']
#Create new knn model
knn2=KNeighborsClassifier() # default =5
#create a dict of all values we want to test for n_neighbors
param_grid={'n_neighbors':np.arange(2,10)}
#Use grid search to test all values from n_neighbors
knn_gscv=GridSearchCV(knn2,param_grid,cv=5)
#fit model
knn_gscv.fit(X,y)
#Check top performing n_neighbors values
# print(help(knn_gscv))
print(knn_gscv.best_params_)
print(knn_gscv.best_score_.round(3))
#Use this optional best n to fit data
knn_final=KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X,y)
y_pred=knn_final.predict(X)
print(knn_final.score(X,y).round(3))
# print(y_pred[:10])
#predict measurement of an iris and record that the L and W of sepal are 5.84, 3.06
#and L and W are 3.76,1.2 use predict()
new_data=np.array([3.76,1.20])
new_data=new_data.reshape(1,-1)# single sample , if( -1,1) : single feature
print(knn_final.predict(np.array(new_data))) 
print(knn_final.predict([[1.45,0.15]]))
# Probability Prediction
new_data2=np.array([[3.76,1.2],[5.25,1.2],[1.58,1.2]])
res2=knn_final.predict(new_data2)
print(res2)
pred2=knn_final.predict_proba(new_data2)
print(pred2)