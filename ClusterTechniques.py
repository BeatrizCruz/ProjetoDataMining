# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:56:49 2019

@author: aSUS
"""

# CLUSTERS
import sqlite3
import os
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler # for the normalization of ConsAff
from sklearn.cluster import KMeans # for the k- means cluster technique
from sklearn.metrics import silhouette_samples # To build the silhouette graph
from sklearn.metrics import silhouette_score   # To build the silhouette graph

# Create this variable: 
# Binary that indicates if a customer gives profit (1) or not (0) to the company:
dfWork['profitBin']=np.where((dfWork['claims']<1,1,0)

# 2 groups of variables: Value/ Engage (costumers) and Consumption/ Affinity (products)
ValueEngage = dfWork[['firstPolicy',
                      'education',
                      'salary',
                      'children',
                      'cmv',
                      'claims',
                      'ratioSalary',
                      'yearCustomer']]

ConsAff = dfWork[['lobMotor',
                  'lobHousehold',
                  'lobHealth',
                  'lobLife',
                  'lobWork',
                  'motorRatio',
                  'householdRatio',
                  'healthRatio',
                  'lifeRatio',
                  'workCRatio',
                  'profitBin']]

# Consumption Normalize (Values between -1 and 1).
# Transform the normalized values into a data frame.
scaler = StandardScaler()
CA_Norm=scaler.fit_transform(ConsAff)
CA_Norm = pd.DataFrame(CA_Norm, columns = ConsAff.columns)

##################################################K-MEANS##############################################################

#------------------------------------------K-means Implementation------------------------------------------------
# 5 initializations k-means.
n_clusters = 3
kmeans = KMeans(n_clusters=3, 
                random_state=0,
                n_init = 5, 
                max_iter = 200).fit(CA_Norm)

# Check the Clusters (Centroids).
my_clusters=kmeans.cluster_centers_
my_clusters

# Invert the transformation for interpretability.
my_clusters=pd.DataFrame(scaler.inverse_transform(X=my_clusters),columns=ConsAff.columns)

# sum of square distances: 
kmeans.inertia_

# Calculate inertia for each number of clusters from 1 to 19 and create a list that will contain the inertias.
L = []
for i in range(1,20):
    kmeans= KMeans(n_clusters=i, random_state=0, n_init=5, max_iter=200).fit(CA_Norm)
    L.append(kmeans.inertia_)

# Creates a plot in which x is between 1 and 19 and y is the inertia values for each cluster number. This plot will show the inertia for each number of clusters through a line chart.
import matplotlib.pyplot as plt
plt.plot(range(1,20),L)

#------------------------------------------Silhouette------------------------------------------------
# Get average silhouette
silhouette_avg = silhouette_score(CA_Norm, kmeans.labels_)
print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(CA_Norm, kmeans.labels_)

cluster_labels = kmeans.labels_

import matplotlib.cm as cm
y_lower = 100

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_ylim([0, CA_Norm.shape[0] + (n_clusters + 1) * 10])

for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i=ith_cluster_silhouette_values. shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color, 
                          alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
plt.show()

######################################################K-modes########################################################
from kmodes.kmodes import KModes
VE_Cat = ValueEngage[['education','livingArea','children','dependents']].astype('str')

for j in list(VE_Cat):
    for i in range(VE_Cat.shape[0]):
        if VE_Cat.loc[i,j] =='':
            VE_Cat.loc[i,j] = 'Missing'

km = KModes(n_clusters=4, init='random', n_init=50, verbose=1)

clusters = km.fit_predict(VE_Cat)

# Print the cluster centroids
print(km.cluster_centroids_)
cat_centroids = pd.DataFrame(km.cluster_centroids_,
                             columns = ['education','status','gender','dependents'])

unique, counts = np.unique(km.labels_, return_counts=True)

cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['Label','Number'])

cat_centroids = pd.concat([cat_centroids, cat_counts], axis = 1)

##################################################Hierarchical Clustering######################################################

# scipy to plot the dendrogram 
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from pylab import rcParams

# The final result will use the sklearn
import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

#Scipy generate dendrograms

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
Z = linkage(CA_Norm,
            method = 'ward')#method='single','complete', 'ward'


# Single: based on distances (nearest points of clusters- closer distance)
# Complete: the farthest points of clusters
# Averege: distances between all the points 
# Default: ward

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
dendrogram(Z,
           truncate_mode='none',
           p=40,
           orientation = 'top',
           leaf_rotation=45,
           leaf_font_size=10,
           show_contracted=True,
           show_leaf_counts=True,color_threshold=50, above_threshold_color='k')

# =============================================================================
# hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k'])
# dendrogram(Z,
#            #truncate_mode='none',
#            truncate_mode='lastp',
#            p=40,
#            orientation = 'top',
#            leaf_rotation=45.,
#            leaf_font_size=10.,
#            show_contracted=True,
#            show_leaf_counts=True, color_threshold=50, above_threshold_color='k')
# =============================================================================

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
#plt.axhline(y=50)
plt.show()

# Aglomerative clustering with scikit learn

#Scikit
k = 3
Hclustering = AgglomerativeClustering(n_clusters = k,
                                      affinity = 'euclidean',
                                      linkage = 'ward' )

#Replace the test with proper data
my_HC = Hclustering.fit(CA_Norm)

my_labels = pd.DataFrame(my_HC.labels_)
my_labels.columns =  ['Labels']
my_labels

# Do the necessary transformations
# Data frame with the centroids

# Create a dataframe that has for each individual the columns normalized and a label column with the cluster each point belongs to:
Affinity = pd.DataFrame(pd.concat([pd.DataFrame(CA_Norm), my_labels],axis=1), 
                        columns=['clothes','kitchen','small_appliances','toys','house_keeping','Labels'])

# Get the centroids for each cluster
to_revert = Affinity.groupby(by='Labels')['clothes','kitchen','small_appliances','toys','house_keeping'].mean()


final_result=pd.DataFrame(scaler.inverse_transform(X=to_revert),
                          columns=['clothes','kitchen','small_appliances','toys','house_keeping'])

####################################################DB-Scan##################################################

from sklearn.cluster import DBSCAN
from sklearn import metrics

db = DBSCAN(eps= 1, #radius (euclidean distance)
            min_samples=10).fit(CA_Norm) # minimum number of points inside the radius.

labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_clusters, counts_clusters = np.unique(db.labels_, return_counts = True)
print(np.asarray((unique_clusters, counts_clusters)))
# 2 clusters when radius is 1
# db scan: might not be the best approach to cluster (all variables are continuous)
# Noise: can be used to find the outliers - blue values : points that are noise are potential outliers.

#use PCA in order to reduce the dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(CA_Norm)
pca_2d = pca.transform(CA_Norm)
for i in range(0, pca_2d.shape[0]):
    if db.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif db.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif db.labels_[i] == 2:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='k',marker='v')
    elif db.labels_[i] == 3:
        c5 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='y',marker='s')
    elif db.labels_[i] == 4:
        c6 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='m',marker='p')
    elif db.labels_[i] == 5:
        c7 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='c',marker='H')
    elif db.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()

#3D

from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(CA_Norm)
pca_3d = pca.transform(CA_Norm)
#Add my visuals
my_color=[]
my_marker=[]
#Load my visuals
for i in range(pca_3d.shape[0]):
    if labels[i] == 0:
        my_color.append('r')
        my_marker.append('+')
    elif labels[i] == 1:
        my_color.append('b')
        my_marker.append('o')
    elif labels[i] == 2:
        my_color.append('g')
        my_marker.append('*')
    elif labels[i] == -1:
        my_color.append('k')
        my_marker.append('<')
        
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(500):
#for i in range(pca_3d.shape[0]):
    ax.scatter(pca_3d[i,0], pca_3d[i,1], pca_3d[i,2], c=my_color[i], marker=my_marker[i])
    
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# Mean Shift: 
# Get the bandwith (is the to eps in dbscan- radio)
# Calculate the mean on each circle (mean data points inside each circle)
# Apply a circle around each mean point
# Repeat the same procedure
# Until find convergence: no more movement

# If you increase the bandwith, there will be a huge cluster and if you decrease it, you will have lots of clusters with few data points in them...- because there are no gaps between points (so this is not a good technique to apply here)
# Continuous variables (with no gaps) are not good for these 2 techniques: clusters formed will not be good.

from sklearn.cluster import MeanShift, estimate_bandwidth

to_MS = CA_Norm
#Instead of guessing the bandwith we can estimate it:
# Calculates the distances between all points: 
# creates a list with all and defines a quantile value that will be used as the bandwith (raio da bandwith)

# The following bandwidth can be automatically detected using
my_bandwidth = estimate_bandwidth(to_MS,
                               quantile=0.2,
                               n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth,
               #bandwidth=0.15, # Bola de cristal: guess the value of the bandwith
               bin_seeding=True)

ms.fit(to_MS)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


#Values
scaler.inverse_transform(X=cluster_centers)

#Count
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)

# lets check our are they distributed

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(to_MS)
pca_2d = pca.transform(to_MS)
for i in range(0, pca_2d.shape[0]):
    if labels[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif labels[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif labels[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Cluster 3 '])
plt.title('Mean Shift found 3 clusters')
plt.show()


#3D
# PCA: to check with less variables the clusters formed
# NOTE: PCA with the variables that are less important and no PCA with the most important.

from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(to_MS)
pca_3d = pca.transform(to_MS)
#Add my visuals
my_color=[]
my_marker=[]
#Load my visuals
for i in range(pca_3d.shape[0]):
    if labels[i] == 0:
        my_color.append('r')
        my_marker.append('+')
    elif labels[i] == 1:
        my_color.append('b')
        my_marker.append('o')
    elif labels[i] == 2:
        my_color.append('g')
        my_marker.append('*')
    elif labels[i] == 3:
        my_color.append('k')
        my_marker.append('<')
        
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#for i in range(pca_3d.shape[0]):
for i in range(250):
    ax.scatter(pca_3d[i,0],
               pca_3d[i,1], 
               pca_3d[i,2], c=my_color[i], marker=my_marker[i])
    
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

################################################## Expectation-Maximization Algorithm  #######################################################

# =============================================================================
# to_EM = my_data[['age','frq','income','rcn']]
# to_EM = to_EM.dropna()
# to_EM.income = to_EM.income.astype(int)
# 
# #test = StandardScaler().fit_transform(test)
# #To reverse
# from sklearn.preprocessing import MinMaxScaler
# my_scaler = MinMaxScaler()
# 
# to_EM_norm = my_scaler.fit_transform(to_EM)
# =============================================================================
from sklearn import mixture
# n_components= <the number of elements you found before- centroids>
gmm = mixture.GaussianMixture(n_components = 3,
                              init_params='kmeans', # {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
                              max_iter=1000,
                              n_init=10,
                              verbose = 1)

gmm.fit(CA_Norm)

# Centroid that has the higher probability assigned to each point.
# Define curves randomly (initialization): random mean and std dev of 1.
# Then define the means and variances with the data.
# How? Likelihood formula applied.

EM_labels_ = gmm.predict(CA_Norm)

# Likelihood value
EM_score_samp = gmm.score_samples(CA_Norm)

# Probabilities of belonging to each cluster.
EM_pred_prob = gmm.predict_proba(CA_Norm)

# Centroids:
scaler.inverse_transform(gmm.means_)

# means_init: probability of each data point being in each cluster formed by another kind of algorithm. 

################################################## SOM ##################################################

# Consumption Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

CA_Norm = scaler.fit_transform(ConsAff)
CA_Norm = pd.DataFrame(CA_Norm, columns = ConsAff.columns)

X = CA_Norm.values



names = ['clothes', 'kitchen', 'small_appliances', 'toys', 'house_keeping']
sm = SOMFactory().build(data = X,
               mapsize=(10,10),
               normalization = 'var',
               initialization='random',#'random', 'pca'
               component_names=names,
               lattice='hexa',#'rect','hexa'
               training ='seq' )#'seq','batch'(look at all the directions and go to a constructed one)

sm.train(n_job=4, #to be faster
         verbose='info', # to show lines when running
         train_rough_len=30, # first 30 steps are big (big approaches) - move 50%
         train_finetune_len=100) # small steps - move 1%

final_clusters = pd.DataFrame(sm._data, columns = ['clothes', 
                                                    'kitchen', 
                                                    'small_appliances', 
                                                    'toys', 
                                                    'house_keeping'])

my_labels = pd.DataFrame(sm._bmu[0]) #100 labels possible (if we have 100 neurons there might be 100 clusters maximum)

final_clusters = pd.concat([final_clusters,my_labels], axis = 1)

final_clusters.columns = ['clothes','kitchen',
                          'small_appliances', 'toys', 
                          'house_keeping', 'Lables']



"""
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.  Options are:
            - gaussian
            - bubble
            - manhattan (not implemented yet)
            - cut_gaussian (not implemented yet)
            - epanechicov (not implemented yet)
        :param normalization: normalizer object calculator. Options are:
            - var
        :param mapsize: tuple/list defining the dimensions of the som.
            If single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som. Options are:
            - planar
            - toroid (not implemented yet)
            - cylinder (not implemented yet)
        :param lattice: type of lattice. Options are:
            - rect
            - hexa
        :param initialization: method to be used for initialization of the som.
            Options are:
            - pca
            - random
        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
"""

from sompy.visualization.mapview import View2DPacked
view2D  = View2DPacked(10,10,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
plt.show()

#These maps' scale is between -1 and 1 because we did the normalization
# Interpretation at the end should be the same
# SOM: less sensitive to outliers

from sompy.visualization.mapview import View2D
view2D  = View2D(10,10,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
plt.show()


# Number of people in each neuron
from sompy.visualization.bmuhits import BmuHitsView
vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="autumn", logaritmic=False)

# Apply k- means over SOM
# K-Means Clustering
from sompy.visualization.hitmap import HitMapView
sm.cluster(3)# <n_clusters>
hits  = HitMapView(10,10,"Clustering",text_size=7)
a=hits.show(sm, labelsize=12)

#Apply hierarchical clustering on the top of SOM

#k means - hierarchical
# Apply first k means with lots of centroids (huge number!!)
# Then, apply hierarchical clustering over the k means centroids

# SOM - K-means?
# SOM: is less sensitive to outliers

# SOM - hierarchical
# Same as we do with k means - hierarchical 










