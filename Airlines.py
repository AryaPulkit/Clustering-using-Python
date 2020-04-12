# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import matplotlib.pylab as plt

os.getcwd()
os.chdir('Desktop/ExcelR/Assignments')
airlines = pd.read_excel("EastWestAirlines.xlsx", sheet_name='data')

#Normalizing_data

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)


# Normalized data frame (considering the numerical part of data)
air_norm = norm_func(airlines.iloc[:,1:])

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch # for creating dendrogram

type(air_norm)

#p = np.array(df_norm) # converting into numpy array format
help(linkage)
dist = linkage(air_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    dist,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)


# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram

from	sklearn.cluster	import	AgglomerativeClustering
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(air_norm)


cluster_labels=pd.Series(h_complete.labels_)

airlines['clust']=cluster_labels # creating a  new column and assigning it to new column
airlines = airlines.iloc[:,[8,0,1,2,3,4,5,6,7]]
airlines.head()

# getting aggregate mean of each cluster
airlines.iloc[:,1:].groupby(airlines.clust).median()

# creating a csv file
airlines.to_csv("airline.csv",encoding="utf-8")
