## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
# import pyrealsense2 as rs
from scipy import spatial
import numpy as np
import cv2
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
from os.path import join
from glob import glob
import pickle
from pprint import pprint as pp
from helpers import get_closest_cloud_point

# Configure depth and color streams
calib_path ='./ros/'


#########################
#     CALIBRATION       #
#########################

# PARAMETERS
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
x_grid = 9
y_grid = 6
images = glob(join(calib_path,'*.jpg'))
cloud = glob(join(calib_path,'*.npy'))
images.sort()
cloud.sort()
objpoints = [] # 3d point in real world space
cloudpoints = [] # 3d cloud points.

images = [images[1]]
cloud = [cloud[1]]
print(cloud)

# PARAMETERS
for iname, cname in zip(images, cloud):
    # load the cloud
    cloudp = np.load(cname)
    cloudp_pixels = cloudp[:,3:]
    cloudp = cloudp[:,:3]
    # Arrays to store object points and image points from all the images.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # We start with the bottom floor
    objp = np.zeros((y_grid,x_grid,3), np.float32)
    for i in range(y_grid):
        for j in range(x_grid):
            objp[i,j,1] = i*2
            objp[i,j,0] = j*2
    # Divide all the points by 100 so they are in meters
    objp = objp/100.0
    # add the height to the second image
    if "_1" in iname:
        objp[:,:,2] = np.ones((y_grid,x_grid))*-0.0334
    # get the color image
    img = cv2.imread(iname)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # get the cloud point

    cv2.imshow('img',img)
    cv2.waitKey(0)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x_grid,y_grid),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("ASDFASDFASDFASDFADSFASD")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        list_corners = corners2.tolist()
        corners_3d = []
        for i in range(len(list_corners)):
            # get pixel
            pixel_img = np.array(list_corners[i][0])
            # for each pixel value find the closest cloud pixel value
            p_index = get_closest_cloud_point(cloudp_pixels, pixel_img)
            cloud_point = cloudp[p_index]
            print(cloud_point)
            corners_3d.append(cloud_point)
        cloudpoints.append(np.array(corners_3d))

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (x_grid,y_grid), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)
# test camera calibration
cloudpoints = np.array(cloudpoints)
# create the homography
cloudpoints = np.array(cloudpoints,dtype='float32').reshape(-1,1,3)
objpoints= np.array(objpoints,dtype='float32').reshape(-1,1,3)
for source, dest in zip(cloudpoints,objpoints):
    print("image :", source)
    print("object:", dest)
    print("//////////////")
H, mask = cv2.findHomography(cloudpoints, objpoints, cv2.RANSAC )
# Save homography
np.savetxt(join(calib_path,"homography.txt"),H)


# Determine error

error = []
for source, dest in zip(cloudpoints,objpoints):
    estimated_dest = np.dot(H,source[0])
    error.append(np.linalg.norm(dest-estimated_dest))
print("AVERAGE EUCLIDEAN ERROR IN METERS:", np.mean(error))
