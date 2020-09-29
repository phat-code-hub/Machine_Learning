import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


data=load_wine()
wine=pd.DataFrame(data.data,columns=data.feature_names)
X=wine[['alcohol','total_phenols']]
scale=StandardScaler()
X_scaled=scale.fit_transform(X)
kmeans=KMeans(n_clusters=3)
kmeans.fit(X_scaled)
y_pred=kmeans.predict(X_scaled)
print(y_pred)
print(kmeans.cluster_centers_)
#Visualize scaled data
plt.scatter(X_scaled[:,0],
            X_scaled[:,1],
            c=y_pred)
#Identify the centroids
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            marker='*',
            s=250, # marker size
            c='red',
            #c=[0,1,2], #== y_pred
            edgecolors='k')
plt.xlabel('alcohol')
plt.ylabel('total_phenols')
plt.title('K-means (k=3)')

#supposed there is new wine (13,2.5)
X_new=np.array([[13,2.5]]) # alcohol=13, total_phenols=2.5
X_new_scaled=scale.transform(X_new)
print(X_new_scaled)
y_new_pred=kmeans.predict(X_new_scaled)
print("Cluster 0= ",kmeans.cluster_centers_[0])
print("new pred = ",y_new_pred) # nam gan cluster 0 :yellow
#plt.xlim(-1.5,-0.5)
plt.show()