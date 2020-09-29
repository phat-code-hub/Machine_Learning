import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
iris=pd.read_csv('../iris.csv')
#build a dict mapping species to integer codes
inv_name_dict={'Iris-setosa':0,
               'Iris-versicolor':1,
               'Iris-virginica':2}
#build integer color code 0,1,2
colors=[inv_name_dict[item] for item in iris['Species']]
#Plot Sepal class
scatter=plt.scatter(iris['SepalLength'],iris['SepalWidth'],c=colors)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal Width(cm)')
plt.legend(handles=scatter.legend_elements()[0],
    labels=inv_name_dict.keys())
plt.show()
#Plot Petal Attribute
scatter=plt.scatter(iris['PetalLength'],iris['PetalWidth'],c=colors)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal Width(cm)')
plt.legend(handles=scatter.legend_elements()[0],
    labels=inv_name_dict.keys())
plt.show()
