# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

cancer_data=load_breast_cancer()
df=pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
df['target']=cancer_data['target']
X=df[cancer_data.feature_names].values
y=cancer_data['target']

param_grid={ 'n_estimators':
    [10,25,50,75,100],
    }


rf=RandomForestClassifier(random_state=123)
gs=GridSearchCV(rf,param_grid,scoring='f1',cv=5)
gs.fit(X,y)
print('best params: ', gs.best_params_)