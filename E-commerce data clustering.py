"""
Created on Wed JULY 2 20:38:13 2020 
1. Clustering Analysis on Data

@author: Muhammad Junaid Hanif
"""


import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pylab as plt
import gower
import seaborn as sns
from sklearn.cluster import KMeans

#%%1. GET THE DATA
dbk = pd.read_csv("epak_1kSKUsBkted.csv")

#%%2. READY THE DATA

cols = dbk.columns
dbk.info()
db = dbk.drop('Basket', axis=1)
cols = db.columns
db.info()

cols_n = db.select_dtypes(include='float64').columns.tolist()
cols_s = [c for c in cols if c not in cols_n]

#%%3. DEALING WITH NAN VALUES
#for cols_n simply replace with 0
db[cols_n] = db[cols_n].fillna(0)
db[cols_s] = db[cols_s].fillna('None')

#%%4. NORMALIZATION USING MINMAXSCALING
mmscaler = preprocessing.MinMaxScaler()
dbn = pd.DataFrame(
                mmscaler.fit_transform(db[cols_n]),
                columns=cols_n)
dbn[cols_s] = db[cols_s]

#%%5. GOWER DISTANCE MATRIX
gmx_dbn = gower.gower_matrix(dbn)
#NOTE: DOES NOT WORK WITH NAN
gmxsq_dbn = squareform(gmx_dbn)


#%%6. HIERARCHICAL CLUSTERING
Z = linkage(gmxsq_dbn, method="ward")


#%%6.1. INERTIA
inertia = [] 
clrange = range(1,15) #Note: 15 means range will stop at 14 as with lists 
for nc in clrange: 
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(Z) 
    inertia.append(kmeans.inertia_ / nc)  
#inertia divided by number of clusters gives us  
#the average squared distance within clusters 
d_inertia = pd.DataFrame({'n_clusters': clrange, 'inertia':inertia}) 
#plt.clf() #only needed if you want clear the plot and try again 
ax = d_inertia.plot(x='n_clusters', y='inertia') 
plt.xlabel('Number of Clusters (k)') 
plt.ylabel('Average within Cluster Squared Distances')

#%%7. DENDROGRAM ANALSYIS
dendrogram(Z,  no_labels=True)
plt.grid(True)
th = 1.8
plt.grid(False)
plt.clf()
dendrogram(Z, color_threshold=th, no_labels=True)
plt.axhline(y=th, c='grey', lw=2, linestyle='dashed')
plt.text(150, th, 't='+str(th))
plt.savefig('dendrogram.png', dpi=300)

#%%8. CLUSTER TAGGING
th = 1.8
db['G1'] = fcluster(Z,t=th,criterion='distance')
dbn['G1'] = fcluster(Z,t=th,criterion='distance')


#%%9. VISUAL POST-CLUSTER ANALYSIS
bkn_sum = dbn.groupby('G1').mean()
dbn.groupby('G1').boxplot(column=cols_n)


plt.clf()
bkn_sum.info()
sns.heatmap(bkn_sum.sort_values(by='Total',ascending=False),
            annot=True,
            cmap='coolwarm')


#%%10. EXPORTED ANALYSIS: PREPARE GROUPBY SUMMARIES FOR EXPORT
bksum = db.groupby('G1')[cols_n].agg(['mean','std'])
bksum['Size'] = db.groupby('G1').size() #We want the number of members in each cluster
bksum[cols_s] = db.groupby('G1')[cols_s].agg(lambda x: x.value_counts().head(2).to_dict())
bksum.to_excel('output/epak_bktsmall_G1Summary.xlsx',sheet_name="Overall")

