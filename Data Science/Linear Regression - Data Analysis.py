from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
boston_dataset=load_boston()
boston=pd.DataFrame(boston_dataset.data,
                    columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target
#print(boston.head())
# print(boston.shape)
# print(boston.columns)
# print(boston.shape[0])
# print(boston[['CHAS','RM','AGE','RAD','MEDV']].head())
# print(boston.describe().round(2))
# print(boston['RM'].describe().round(2))
# #boston['CHAS'].hist()
# boston.hist(column='CHAS',color='r',label='CHARS')
# # plt.title('CHARS')
# # plt.show()
# # boston.hist(column='RM',color='b',bins=20)
# # plt.style.use('ggplot')
# # plt.title('RM')
# # plt.show()
# print(boston['RM'].describe().round(2))
# boston['RM'].plot.hist(title='Room',bins=25)
boston['RM'].plot.hist(title='Tuan',bins=20)