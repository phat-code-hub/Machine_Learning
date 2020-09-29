# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data=load_breast_cancer()
df=pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
df['target']=cancer_data['target']
X=df[cancer_data.feature_names].values
y=cancer_data['target']
print('data dimension',X.shape)