import numpy as np
import csv
import scipy
import pickle as pkl
from helpers import parse_input_data, load_pose_by_desc
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load the data for the left arm
l_arm_data = parse_input_data("./data/Yumi/","./data/Yumi/task_params.csv",True,"left")
arm = "left"
surgeme = 1
# Filter it to surgeme 1 that stats with the left arm pickup
s1_data = l_arm_data[(l_arm_data[:,12]==surgeme)&(l_arm_data[:,13]<7)]
print s1_data.shape
# Rebase by target
# subtract the n peg by with rotation x with peg 2 with rotation 1 (base peg)
# Load Peg positions
pose_matrix = []
for rot in range(1,4):
    rot_poses = []
    for peg in range(1,7):
        pos = load_pose_by_desc('left',peg,rot)
        rot_poses.append(pos)
    pose_matrix.append(rot_poses)
# finised Peg position loading
# This vector is then subtracted to all the points in the sample.
# and saved in rebased data. Rebased data saves all the data segments

rebased_data = []
surgeme = []
prev_peg = -1
prev_rot = -1
first = True
for elem in s1_data:
    row = []
    # get the rebasing translation
    elem_rot = int(elem[14])
    elem_peg = int(elem[13])
    if (elem_peg != prev_peg or elem_rot != prev_rot) and not first:
        # surgeme finished
        rebased_data.append(surgeme)
        surgeme = []
    offset = pose_matrix[elem_rot-1][elem_peg-1][arm].translation-pose_matrix[0][0][arm].translation
    rebased_pos = elem[0:3]-offset
    row += list(rebased_pos)
    row += list(elem[12:])
    surgeme.append(row)
    prev_peg = elem_peg
    prev_rot = elem_rot
    first = False

#For each surgeme, get the splines (keep target peg, label and rotation)
final_data = []
rebased_index = 0
for elem in rebased_data:
    surgeme = np.array(elem)
    surgeme_pos = surgeme[:,:3]
    # Eliminate the points that are repeated, since the interpolation
    # that estimates the spline does not accept them
    _,x_index = np.unique(surgeme[:,0],return_index = True)
    _,y_index = np.unique(surgeme[:,1],return_index = True)
    _,z_index = np.unique(surgeme[:,2],return_index = True)
    xyz_index = np.concatenate((x_index,y_index,z_index),axis = None)
    xyz_index = np.sort(np.unique(xyz_index))

    unique_pos = surgeme_pos[xyz_index]
    tck, u = interpolate.splprep([
        unique_pos[:,0],
        unique_pos[:,1],
        unique_pos[:,2]],s = 3)
    # Create a new data point
    # add start point

    data_row = []
    data_row.extend(list(surgeme[0,:3]))
    # add target
    data_row.extend(pose_matrix[0][0][arm].translation)
    # add tck points
    knots, coeff, degree = tck
    # add the 12 coefficients that are going to be predicted
    data_row.extend(list(np.array(coeff).reshape((12,))))
    # EXTRA PLOTTING INFO
    # add the spline knots
    data_row.extend(list(knots))
    # add the label, peg and rotation
    data_row.extend(list(surgeme[0,3:]))
    data_row.append(rebased_index)
    final_data.append(data_row)
final_data = np.array(final_data)

# Train
# get training and testing splits
x_full = final_data[:,:6]
y_full = final_data[:,6:18]
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=0.2, random_state=42)
# y_train = y_train.reshape((-1,1))
# y_test = y_test.reshape((-1,1))
reg = LinearRegression().fit(x_train, y_train)
print "REGRESSION SCORE", reg.score(x_train, y_train)


# TRY WITH A NN_REGRESSION
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,12)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()

# Normalization and preprocessing here
# Make the data into tensors
train_tensor = torch.tensor(np.concatenate((x_train,y_train), axis=1)).float()
test_tensor = torch.tensor(np.concatenate((x_test,y_test), axis=1)).float()

for epoch in range(100):
    for i, data2 in enumerate(train_tensor):
        X=data2[0:6]
        Y=data2[6:18]
        X, Y = Variable(X, requires_grad=True), Variable(Y, requires_grad=False)
        optimizer.zero_grad()
        y_pred = net(X)
        output = criterion(y_pred, Y)
        output.backward()
        optimizer.step()
    if (epoch % 20 == 0.0):
        print("Epoch {} - loss: {}".format(epoch, output))


