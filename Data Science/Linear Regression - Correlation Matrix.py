from sklearn.datasets import load_boston
import pandas as pd
# import matplotlib.pyplot as plt
boston_dataset=load_boston()
boston=pd.DataFrame(boston_dataset.data,
                    columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target
corr_mat=boston.corr().round(2)
print(corr_mat)
boston.plot(kind='scatter',
            x='RM',
            y='MEDV',
            figsize=(8,6))
boston.plot(kind='scatter',
            x='LSTAT',
            y='MEDV',
            figsize=(8,6))
X=boston[['RM']]
print(X.shape)
y=boston['MEDV']
print(y.shape)
