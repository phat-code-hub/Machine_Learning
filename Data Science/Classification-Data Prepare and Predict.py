import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris=pd.read_csv('../iris.csv')
X=iris[['PetalLength','PetalWidth']]
y=iris['Species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=1,stratify=y)
# print(y_train.value_counts())
# print(y_test.value_counts())
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
# print(X_test)
# print(pred[:5])
#Probability Prediction
y_pred_prob=knn.predict_proba(X_test)
print(y_pred_prob[:12]) #ndarray
print(pred[:12])
