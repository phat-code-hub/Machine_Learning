import pandas as pd
import matplotlib.pyplot as plt
iris=pd.read_csv('../iris.csv')
print(iris.shape)
print(iris.head())
iris.drop('id',axis=1,inplace=True)
# print(iris.shape)
# print(iris.head())
# print(iris.describe().round(2))
print(iris.groupby('Species').size())
print(iris['Species'].value_counts()) # Same result with above code
iris.hist(color='r')
plt.show()