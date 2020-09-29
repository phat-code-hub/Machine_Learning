# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data=load_breast_cancer()
print(cancer_data.keys())
# print(cancer_data.DESCR)
print(cancer_data['data'].shape)
# print(cancer_data['data'][:20,:4])
print(cancer_data['feature_names'])
df=pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
print(df.head())
print(cancer_data['target'].shape)
print(cancer_data['target_names'])
df['target']=cancer_data['target']
df.head()