import numpy as np
import csv
import scipy
import picke as pkl

data_x = np.loadtxt("./data/yumi_kinematics_feature_x.txt",dtype=np.float)
data_y = np.loadtxt("./data/yumi_kinematics_feature_y.txt",dtype=np.float)

s1 = data_x[84,4:7]
target = pkl_file = open('pose.pkl', 'rb')


