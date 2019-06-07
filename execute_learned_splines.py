import numpy as np
import csv
import scipy
import pickle as pkl
from helpers import *
import matplotlib
matplotlib.use('TkAgg')
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
import pickle as pkl

from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import pickle as pkl
import csv

##########################
###     PARAMS         ###
##########################

# Load the data for the left arm
end_surgeme = 3
models = []
for i in range(1,end_surgeme+1):
    if save_model:
        with open('models/S'+str(surgeme_number)+'_regression', 'rb') as model_name:
            reg = pkl.dump(model_name)
            models.append(reg)

# init robot
y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
y.left.set_tool(DELTALEFT)
y.right.set_tool(DELTARIGHT)
y.set_v(40)
y.set_z('z100')
# close arms
y.left.close_gripper(force=2,wait_for_res=False)
y.right.close_gripper(force=2,wait_for_res=False)
# do approach
# load init position
s1_init = load_pose_by_path("data/s1_end_l_1")["left"]
# go to the init pose
y.left.goto_pose(s1_init,False,True,False)
# Create inputs
inputs = [s1_init.traslation]
# Load inputs
s1_end = load_pose_by_desc("left",1,1)["left"]
s1.translation[3] = s1.translation[3] + 0.01
inputs.append(s1_end.translation)
# predict the path
pred_waypoints = reg.predict(inputs)

# Get the predicted spline
# add the initial point
x_way = []
y_way = []
z_way = []
x_way.append(orig_data[0,0])
y_way.append(orig_data[0,1])
z_way.append(orig_data[0,2])
# add the waypoints
pred_waypoints = pred_waypoints.reshape(6)
for i in range(coeff_len/3):
    x_way.append(pred_waypoints[i*3])
    y_way.append(pred_waypoints[i*3 +1])
    z_way.append(pred_waypoints[i*3 +2])
# add the endpoint
x_way.append(orig_data[-1,0])
y_way.append(orig_data[-1,1])
z_way.append(orig_data[-1,2])
# get the spline
t_points = np.linspace(0,1,30)
tck_way, u_way = interpolate.splprep([x_way,y_way,z_way ], s=spline_degree)
x_pred, y_pred, z_pred = interpolate.splev(t_points, tck_way)

go_to_poses = []
for point in predicted_points:
	target_pose = copy.deepcopy(s1_init)
	target_pose.translation=point[:3]
	target_pose.rotation=RigidTransform.rotation_from_quaternion(point[3:7])
	go_to_poses.append(target_pose)
y.left.buffer_clear()
y.left.buffer_add_all(go_to_poses)
y.set_v(30)
#self.yumi.set_z('z100')
y.left.buffer_move()


# execute the path

# align and grasp
