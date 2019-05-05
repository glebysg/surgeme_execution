# -*- coding: utf-8 -*-
"""
Created on Sat May  4 01:56:42 2019

@author: jyoth
"""
try:
    del data_x, data_y
except:
    pass

import numpy as np
import math
import scipy.optimize as optimize
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.interpolate import interp1d

def twoD_Gauss(data,amplitude,x0,y0,sigma_x,sigma_y,offset):
    x=data[:,0]
    y=data[:,1]
    x0=float(x0)
    y0=float(y0)
    return offset + amplitude*np.exp(-(((x-x0)**(2)/(2*sigma_x**(2))) + ((y-y0)**(2)/(2*sigma_y**(2)))))

def Gauss(x,amplitude,x0,sigma_x,offset):
    x0=float(x0)
    return offset + amplitude*np.exp(-(((x-x0)**(2)/(2*sigma_x**(2)))))

data = np.loadtxt(r'C:\Users\jyoth\OneDrive\Desktop\data.txt',dtype = float)
labels = np.loadtxt(r'C:\Users\jyoth\OneDrive\Desktop\labels.txt',dtype = float)
unified = np.column_stack((data,labels))
print(unified.shape)

a = np.split(unified, np.where(np.diff(unified[:,14]) != 0)[0]+1)
len_a = len(a)
print(len_a)

l = []
for i in range(len(a)):
    l.append(a[i].shape[0])
max_val_l = max(l)

b = []
for i in range(max_val_l):
    b.append(sum(abs(np.array(l) - i)))
best_val_l = b.index(min(b))

d = []
#degree = int(best_val_l/2)
#no_of_param = 3*(degree+1)
no_of_comp = 1
if no_of_comp > 1:
    gmm_param = 7*no_of_comp
else:
    gmm_param = 6*no_of_comp
data_x = np.zeros((len_a,1,gmm_param))
data_y = np.zeros((len_a,1,7)) 
#interm = np.zeros((len_a,best_val_l,3)) 
for i in range(len_a):
        data_y[i,0,0] = int(a[i][0,14])
        data_y[i,0,1:4] = a[i][0,0:3]
        data_y[i,0,4:7] = a[i][-1,0:3]
        #params, pcov = optimize.curve_fit(twoD_Gauss, a[i][:,0:2], a[i][:,2])
        gmm = GaussianMixture(n_components=no_of_comp, covariance_type='diag')
        gmm.fit(a[i][:,0:3])
        if no_of_comp > 1:
            data_x[i,0,:] = np.hstack((np.asarray(gmm.weights_).reshape(-1),np.asarray(gmm.means_).reshape(-1),np.asarray(gmm.covariances_).reshape(-1)))
        else:
            data_x[i,0,:] = np.hstack((np.asarray(gmm.means_).reshape(-1),np.asarray(gmm.covariances_).reshape(-1)))         
        d.append(gmm.converged_)
        #data_x[i,0,:] = params
#        t = np.arange(a[i].shape[0])
#        print(i)
#        params, pcov = optimize.curve_fit(Gauss, t, a[i][:,1])
#        data_x[i,0,:] = params
#        if a[i].shape[0] <= best_val_l:
#            for j in range(best_val_l):
#                try:
#                    interm[i,j,:] = a[i][j][0:3]
#                    k = a[i][j][0:3]
#                except IndexError:
#                    interm[i,j,:] = k 
#        else:
#            length = float(a[i].shape[0])
#            num = best_val_l - 2
#            for j in range(best_val_l):
#                interm[i,0,:] = a[i][0][0:3]
#                interm[i,best_val_l-1,:] = a[i][-1][0:3]
#            for k in range(1,best_val_l-1):
#                interm[i,k,:] = a[i][int(math.ceil((k-1)*length/num))][0:3]  
#        t = np.arange(best_val_l)
#        fitx = np.polyfit(t,interm[i,:,0],degree)
#        fity = np.polyfit(t,interm[i,:,1],degree)
#        fitz = np.polyfit(t,interm[i,:,1],degree)
#        data_x[i,0,:] = np.hstack((fitx,fity,fitz))
                
if len(np.unique(d))>1:
    print('gmm convergence problem')

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(None,7)))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add((Dense(gmm_param,activation='linear')))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(data_y, data_x,epochs=100,batch_size=40,validation_split=0.2)

y_test = data_y[0:7]
#print('predicted')
#print(model.predict(y_test))
#print('actual')
#print(data_x[0:7])
predicted1 = model.predict(y_test)
#actual1 = a[0][:,0:3]

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

for i in range(7):
        gmm_new = GaussianMixture(n_components=no_of_comp, covariance_type='diag')
        gmm_new.fit(np.random.rand(len(a[i]),3))
        if no_of_comp > 1:
            gmm_new.weights_ = np.asarray([predicted1[i,0,0]]) 
        else:
            pass
        gmm_new.means_ = np.asarray([predicted1[i,0,0:int(gmm_param/2)]])
        gmm_new.covariances_ = np.asarray([predicted1[i,0,int(gmm_param/2):int(gmm_param)]])
        x = gmm_new.sample(len(a[i]))
        print(x[0].shape)
        x[0][:,0] = [a[i][0,0] if math.isnan(k) else k for k in x[0][:,0]]
        x[0][:,1] = [a[i][0,1] if math.isnan(k) else k for k in x[0][:,1]]
        x[0][:,2] = [a[i][0,2] if math.isnan(k) else k for k in x[0][:,2]]
        print(x[0].shape)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xliner = x[0][:,0]
        yliner = x[0][:,1]
        zliner = x[0][:,2]
        ax.plot3D(xliner, yliner, zliner, 'gray')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xlinel = a[i][:,0]
        ylinel = a[i][:,1]
        zlinel = a[i][:,2]
        ax.plot3D(xlinel, ylinel, zlinel, 'blue')
        del gmm_new

#for i in range(7):
#    for j in range(7):
#        fig = plt.figure(int(100+i*10+j))
#        line = predicted1[i,:,j]
#        ax = plt.subplot(111)
#        ax.plot(line)
#        newline = actual1[i,:,j]
#        ax.plot(newline)
#        string = 'plot1'
#        string += str(i*10+j)
#        string += '.png'
#        fig.savefig(string)
        
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#for i in range(7):
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    t = np.arange(a[i].shape[0])
#    xliner = np.polyval(predicted1[i,0,0:degree+1],t)
#    yliner = np.polyval(predicted1[i,0,degree+1:2*(degree+1)],t)
#    zliner = np.polyval(predicted1[i,0,2*(degree+1):3*(degree+1)],t)
#    ax.plot3D(xliner, yliner, zliner, 'gray')
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    xlinel = np.polyval(data_x[i,0,0:degree+1],t)
#    ylinel = np.polyval(data_x[i,0,degree+1:2*(degree+1)],t)
#    zlinel = np.polyval(data_x[i,0,2*(degree+1):3*(degree+1)],t)
#    ax.plot3D(xlinel, ylinel, zlinel, 'blue')






