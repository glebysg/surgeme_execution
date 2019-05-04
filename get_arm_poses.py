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
import IPython
import argparse
import pickle as pkl

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action="store", dest="arm", default="left",
            help="record arm. Possible options: left, right or both")
    parser.add_argument('-s', action="store", dest="filename", default="pose",
            help="name of the pkl file where the object is recorded")
    args = parser.parse_args()
    poses = {'left': None, 'right':None}
    if args.arm == "left" or args.arm == "both":
    poses['left']=y.left.get_pose()
    if args.arm == "right" or args.arm == "both":
    	poses['right']=y.left.get_pose()
    # save pkl object
    parser.parse_args()
    with open(parser.args['filename'], "wb") as output_file:
        pkl.dump(poses,output_file)

if __name__ == '__main__':
    main()

