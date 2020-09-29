import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
#from IPython.display import Image

df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
feature_names=['Pclass','male']
X=df[feature_names]
y=df['Survived']

dt=DecisionTreeClassifier()
dt.fit(X,y)

dot_file=export_graphviz(dt,feature_names=feature_names)
graph=graphviz.Source(dot_file)
graph.render(filename='Decision Tree',format='png',cleanup=True)