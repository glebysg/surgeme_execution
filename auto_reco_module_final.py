# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:26:53 2019

@author: jyoth
"""

import numpy as np
import pandas as pd

features = np.loadtxt(r'C:\Users\jyoth\OneDrive\Desktop\yumi_data.txt',dtype = float)
labels = np.loadtxt(r'C:\Users\jyoth\OneDrive\Desktop\yumi_labels.txt',dtype = float)
unified = np.column_stack((features,labels))

data = np.split(unified, np.where(np.diff(unified[:,14]) != 0)[0]+1)

data_order = []
for i in range(len(data)):
    if int(data[i][0,14]) == 1:
        data_order.append(data[i])
for i in range(len(data)):
    if int(data[i][0,14]) == 2:
        data_order.append(data[i])
for i in range(len(data)):
    if int(data[i][0,14]) == 3:
        data_order.append(data[i])
for i in range(len(data)):
    if int(data[i][0,14]) == 4:
        data_order.append(data[i])
for i in range(len(data)):
    if int(data[i][0,14]) == 5:
        data_order.append(data[i])
for i in range(len(data)):
    if int(data[i][0,14]) == 6:
        data_order.append(data[i])
for i in range(len(data)):
    if int(data[i][0,14]) == 7:
        data_order.append(data[i])
data_order_copy = data_order
        
data_var = np.zeros((len(data),14))
for i in range(len(data)):
    for j in range(14):
        data_var[i,j]=np.std(data_order[i][:,j])

ch_pts = []        
for i in range(len(data_order)-1):
    if np.unique(data_order[i+1][:,14])-np.unique(data_order[i][:,14]) == 1:
        ch_pts.append(i)

               
data_var_mean = np.zeros((7,14))
data_var_mean[0] = np.mean(data_var[:ch_pts[0]+1],axis = 0)
data_var_mean[1] = np.mean(data_var[ch_pts[0]+1:ch_pts[1]+1],axis = 0)
data_var_mean[2] = np.mean(data_var[ch_pts[1]+1:ch_pts[2]+1],axis = 0)
data_var_mean[3] = np.mean(data_var[ch_pts[2]+1:ch_pts[3]+1],axis = 0)
data_var_mean[4] = np.mean(data_var[ch_pts[3]+1:ch_pts[4]+1],axis = 0)
data_var_mean[5] = np.mean(data_var[ch_pts[4]+1:ch_pts[5]+1],axis = 0)
data_var_mean[6] = np.mean(data_var[ch_pts[5]+1:],axis = 0)
   
cor_surgeme = np.zeros((7,120))
        
df = pd.DataFrame(index = list('xyzrptgXYZRPTGs'), columns=list('xyzrptgXYZRPTGs')).fillna(0)
for i in range(ch_pts[0]+1):
    df = df + pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
df = df/(ch_pts[0]+1)
cor_surgeme[0] = df.values[np.triu_indices(15,k=0)]
        
df = pd.DataFrame(index = list('xyzrptgXYZRPTGs'), columns=list('xyzrptgXYZRPTGs')).fillna(0)
for i in range(ch_pts[0]+1,ch_pts[1]+1):
    df = df + pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
df = df/(ch_pts[1]-ch_pts[0])
cor_surgeme[1] = df.values[np.triu_indices(15,k=0)]

df = pd.DataFrame(index = list('xyzrptgXYZRPTGs'), columns=list('xyzrptgXYZRPTGs')).fillna(0)
for i in range(ch_pts[1]+1,ch_pts[2]+1):
    df = df + pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
df = df/(ch_pts[2]-ch_pts[1])
cor_surgeme[2] = df.values[np.triu_indices(15,k=0)]

df = pd.DataFrame(index = list('xyzrptgXYZRPTGs'), columns=list('xyzrptgXYZRPTGs')).fillna(0)
for i in range(ch_pts[2]+1,ch_pts[3]+1):
    df = df + pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
df = df/(ch_pts[3]-ch_pts[2])
cor_surgeme[3] = df.values[np.triu_indices(15,k=0)]

df = pd.DataFrame(index = list('xyzrptgXYZRPTGs'), columns=list('xyzrptgXYZRPTGs')).fillna(0)
for i in range(ch_pts[3]+1,ch_pts[4]+1):
    df = df + pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
df = df/(ch_pts[4]-ch_pts[3])
cor_surgeme[4] = df.values[np.triu_indices(15,k=0)]

df = pd.DataFrame(index = list('xyzrptgXYZRPTGs'), columns=list('xyzrptgXYZRPTGs')).fillna(0)
for i in range(ch_pts[4]+1,ch_pts[5]+1):
    df = df + pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
df = df/(ch_pts[5]-ch_pts[4])
cor_surgeme[5] = df.values[np.triu_indices(15,k=0)]

df = pd.DataFrame(index = list('xyzrptgXYZRPTGs'), columns=list('xyzrptgXYZRPTGs')).fillna(0)
for i in range(ch_pts[5]+1,len(data)):
    df = df + pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
df = df/(len(data)-ch_pts[5]-1)
cor_surgeme[6] = df.values[np.triu_indices(15,k=0)]

cor_surgeme = np.nan_to_num(cor_surgeme)
#cor_surgeme = (abs(np.round(cor_surgeme))==1).astype(int)

cluster_data = np.concatenate((data_var_mean, cor_surgeme), axis=1)

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

gmm = GaussianMixture(n_components=2)
gmm.fit(cluster_data)
print(np.round(gmm.predict_proba(cluster_data)))

km = KMeans(n_clusters=2)
km.fit(cluster_data)
print(km.labels_)

i = 0
test_df = pd.DataFrame(data_order[i], columns=list('xyzrptgXYZRPTGs')).corr().fillna(0)
test_data = np.concatenate((data_var[i].reshape(1,-1), np.nan_to_num(test_df.values[np.triu_indices(15,k=0)]).reshape(1,-1)), axis=1)
print(np.round(gmm.predict_proba(test_data)))
print(km.predict(test_data))
