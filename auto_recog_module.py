# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:32:55 2019

@author: jyoth
"""

import numpy as np

data = np.loadtxt(r'C:\Users\jyoth\OneDrive\Desktop\yumi_data.txt',dtype = float)
labels = np.loadtxt(r'C:\Users\jyoth\OneDrive\Desktop\yumi_labels.txt',dtype = float)
unified = np.column_stack((data,labels))

a = np.split(unified, np.where(np.diff(unified[:,14]) != 0)[0]+1)

new_a = []
for i in range(len(a)):
    if int(a[i][0,14]) == 1:
        new_a.append(a[i])
for i in range(len(a)):
    if int(a[i][0,14]) == 2:
        new_a.append(a[i])
for i in range(len(a)):
    if int(a[i][0,14]) == 3:
        new_a.append(a[i])
for i in range(len(a)):
    if int(a[i][0,14]) == 4:
        new_a.append(a[i])
for i in range(len(a)):
    if int(a[i][0,14]) == 5:
        new_a.append(a[i])
for i in range(len(a)):
    if int(a[i][0,14]) == 6:
        new_a.append(a[i])
for i in range(len(a)):
    if int(a[i][0,14]) == 7:
        new_a.append(a[i])
        
data = np.zeros((len(a),14))
for i in range(len(a)):
    for j in range(14):
        data[i,j]=np.std(new_a[i][:,j])

ch_pts = []        
for i in range(len(new_a)-1):
    if np.unique(new_a[i+1][:,14])-np.unique(new_a[i][:,14]) == 1:
        ch_pts.append(i)
        
data_new = np.zeros((len(a),3))
data_new[:,0] =  data[:,0] + data[:,1] + data[:,2] + data[:,7] + data[:,8] + data[:,9]
data_new[:,1] =  data[:,3] + data[:,4] + data[:,5] + data[:,10] + data[:,11] + data[:,12]
data_new[:,2] =  data[:,6] + data[:,13]

lab = np.zeros((len(a),1))
for i in range(len(new_a)):
    if data_new[i,0] > data_new[i,1] or data_new[i,0] > data_new[i,2]:
        lab[i] = 1
data = lab
        
data_new = np.zeros((7,1))
data_new[0] = np.mean(data[:ch_pts[0]+1],axis = 0)
data_new[1] = np.mean(data[ch_pts[0]+1:ch_pts[1]+1],axis = 0)
data_new[2] = np.mean(data[ch_pts[1]+1:ch_pts[2]+1],axis = 0)
data_new[3] = np.mean(data[ch_pts[2]+1:ch_pts[3]+1],axis = 0)
data_new[4] = np.mean(data[ch_pts[3]+1:ch_pts[4]+1],axis = 0)
data_new[5] = np.mean(data[ch_pts[4]+1:ch_pts[5]+1],axis = 0)
data_new[6] = np.mean (data[ch_pts[5]+1:],axis = 0)
data = np.round(data_new)
   
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)
gmm.fit(data)
print(gmm.predict_proba(data))

i =  0 #test example, give any number
data = np.zeros((1,14))
for j in range(14):
    data[0,j]=np.std(new_a[i][:,j])
data_new = np.zeros((1,3))
data_new[0,0] =  data[0,0] + data[0,1] + data[0,2] + data[0,7] + data[0,8] + data[0,9]
data_new[0,1] =  data[0,3] + data[0,4] + data[0,5] + data[0,10] + data[0,11] + data[0,12]
data_new[0,2] =  data[0,6] + data[0,13]
lab = 0
if data_new[0,0] > data_new[0,1] or data_new[0,0] > data_new[0,2]:
        lab = 1
data = np.asarray(lab).reshape(1,-1)

print(gmm.predict_proba(data))

