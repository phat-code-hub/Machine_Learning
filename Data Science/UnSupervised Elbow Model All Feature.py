import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


data=load_wine()
wine=pd.DataFrame(data.data,columns=data.feature_names)
X=wine
scale=StandardScaler()
X_scaled=scale.fit_transform(X)
#Visualize scaled data 
inertias=[]
sum_dif=[]
first=0
sum=0
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sum_dif.append(km.inertia_-first)
    first=km.inertia_
    
print("Inertia: ",inertias)
print("DIFFER: ",sum_dif)
#Plot inertia
plt.plot(np.arange(1,11),inertias,marker='o',color='purple')
plt.plot([3,3],[0,2500],linestyle='--',color='r',linewidth=3)
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.title('All Features')
plt.show()
