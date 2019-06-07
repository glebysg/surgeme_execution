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
import copy
import time

##########################
###     PARAMS         ###
##########################

# Load the data for the left arm
spline_degree = 3
end_surgeme = 3
coeff_len = 6
peg_n = 6
models = []
for i in range(1,end_surgeme+1):
    with open('models/S'+str(i)+'_regression', 'rb') as model_name:
        reg = pkl.load(model_name)
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
# Create inputs
s_init =  None
s_end =  None
for surgeme_number in range(1,end_surgeme+1):
    # load init and end position
    if surgeme_number == 1:
        # Load inputs
        s_init = load_pose_by_path("poses/s1_init_l_6")["left"]
        s_end = load_pose_by_desc("left",peg_n,1)["left"]
        s_end.translation[2] = s_end.translation[2] + 0.001
    elif surgeme_number == 2:
        s_init = y.left.get_pose()
        s_end = load_pose_by_path("poses/s2_end_l_"+str(peg_n))["left"]
        s_end.translation[2] = s_end.translation[2] + 0.001
    elif surgeme_number == 3:
        s_init = y.left.get_pose()
        s_end = load_pose_by_path("poses/s3_end_l_"+str(peg_n))["left"]

    # go to the init pose
    y.left.goto_pose(s_init,False,True,False)
    inputs = [s_init.translation]
    inputs.append(s_end.translation)
    inputs = np.array(inputs).reshape(1,-1)
    # predict the path
    pred_waypoints = reg.predict(inputs)

    # Get the predicted spline
    # add the initial point
    x_way = []
    y_way = []
    z_way = []
    x_way.append(s_init.translation[0])
    y_way.append(s_init.translation[1])
    z_way.append(s_init.translation[2])
    # add the waypoints
    pred_waypoints = pred_waypoints.reshape(coeff_len)
    for i in range(coeff_len/3):
        x_way.append(pred_waypoints[i*3])
        y_way.append(pred_waypoints[i*3 +1])
        z_way.append(pred_waypoints[i*3 +2])
    # add the endpoint
    x_way.append(s_end.translation[0])
    y_way.append(s_end.translation[1])
    z_way.append(s_end.translation[2])
    # get the spline
    t_points = np.linspace(0,1,20)
    tck_way, u_way = interpolate.splprep([x_way,y_way,z_way ], s=spline_degree)
    x_pred, y_pred, z_pred = interpolate.splev(t_points, tck_way)

    go_to_poses = []
    count = 0
    for x_coord, y_coord, z_coord in zip(x_pred, y_pred, z_pred):
            target_pose = copy.deepcopy(s_init)
            target_pose.translation[0] = x_coord
            target_pose.translation[1] = y_coord
            target_pose.translation[2] = z_coord
            y.left.goto_pose(target_pose,False,True,False)
    if surgeme_number == 2:
        y.left.move_gripper(0.005)
        y.left.close_gripper(force=2,wait_for_res=False)
    time.sleep(3)


	# go_to_poses.append(target_pose)
# print s_init
# print go_to_poses
# y.left.buffer_clear()
# y.left.buffer_add_all(go_to_poses)
# y.set_v(30)
# self.yumi.set_z('z100')
# y.left.buffer_move(wait_for_res=False)


# execute the path

# align and grasp
