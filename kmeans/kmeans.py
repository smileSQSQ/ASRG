# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate 
@File: kmeans.py
@Author: QI
@Date: 2021/9/28 2:24 
@Description: None
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pylab as pl
from sklearn.decomposition import PCA

# Plot styling
import seaborn as sns

sns.set()  # for plot styling

mpl.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('ggplot')

dataset = pd.read_csv('CLV.csv')

Income = dataset['INCOME'].values
Spend = dataset['SPEND'].values
X = np.array(list(zip(Income, Spend)))
plt.scatter(Income, Spend, c='black', s=100)
plt.xlabel(u"聚类结果(iter=0)")  # X轴标签

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:, 0], X[:, 1])

plt.show()

#########################

X = dataset.iloc[:, [0, 1]].values

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)

# Calculating the silhoutte coefficient
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

for k in range(1, 11):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_
    interia = kmeans_model.inertia_
    print("k:", k, " cost:", interia)
print()

##Fitting kmeans to the dataset
km4 = KMeans(n_clusters=3, init='k-means++', max_iter=11, n_init=10, random_state=0)
y_means = km4.fit_predict(X)

# Visualising the clusters for k=4
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='purple', label=u'类别1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='blue', label=u'类别2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='green', label=u'类别3')

plt.scatter(km4.cluster_centers_[:, 0], km4.cluster_centers_[:, 1], s=200, marker='s', c='red', alpha=0.7, label=u'聚类中心')
plt.xlabel(u'聚类结果(iter=50)')
plt.legend()
plt.show()
