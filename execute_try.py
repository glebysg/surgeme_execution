import logging
import time
import os
import unittest
import numpy as np
import copy
import sys

from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import csv
import IPython

y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)

#setup the ool distance for the surgical grippers
# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])

DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
y.left.set_tool(DELTALEFT)
y.right.set_tool(DELTARIGHT)
y.set_v(40)
y.set_z('z100')

# def test_fast_pose_bufer(self):
#         start = time.time()
#         all_poses = self._get_circular_poses()
#         self.yumi.left.buffer_clear()
#         self.yumi.left.buffer_add_all(all_poses)
#         self.yumi.set_v(800)
#         self.yumi.set_z('z100')
#         self.yumi.left.buffer_move()
#         end = time.time()
#         dur = end - start

#OPEN AND CLOSE GRIPPER for the surgical gripper


def left_close(y):
    y.left.close_gripper(force=2,wait_for_res=False)
    
def left_open(y):
    y.left.move_gripper(0.005)
    
def right_close(y):
    y.right.close_gripper(force=2,wait_for_res=False)

def right_open(y):
    y.right.move_gripper(0.005)

# rotation_target=RigidTransform.rotation_from_quaternion(quaternion_target)
# Initialize arms
init_pose_left=y.left.get_pose()
init_pose_right=y.right.get_pose()
# init_pose_left.translation = [0.3968,0.06525,-0.04631]
# init_pose_right.translation = [0.3969,0.06525,-0.0788]
# Go to the initial possition
y.left.goto_pose(init_pose_left,False,True,False)
predicted_points = np.loadtxt('final_output_predicted_left.txt', delimiter=',')
go_to_poses = []
for point in predicted_points:
	target_pose = copy.deepcopy(init_pose_left)
	target_pose.translation=point[:3]
	target_pose.rotation=RigidTransform.rotation_from_quaternion(point[3:7])
	go_to_poses.append(target_pose)
y.left.buffer_clear()
y.left.buffer_add_all(go_to_poses)
y.set_v(30)
#self.yumi.set_z('z100')
y.left.buffer_move()

	# # read file poses for execution:
	# #init_pose_right=y.right.get_pose()
	# target_left=copy.deepcopy(init_pose_left)
	# delta=[-0.01,0,0]
	# target_left.translation=target_left.translation+delta
	# translation=[0.01,0,0]
	# # y.left.goto_pose_delta(translation, rotation=None, wait_for_res=True)

	# #Function used is: goto_pose(pose, linear=True, relative=False, wait_for_res=True)
	# y.left.goto_pose(target_left,False,True,False)

	# time.sleep(0.2)	
	# left_close(y)
	# left_open(y)

# print "pose is",a
