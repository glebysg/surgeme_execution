## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
from os.path import join
from glob import glob
import pickle

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
count = 0
img_path ='realsense_calibration'
save_img=False
try:
    #########################
    #     DATA COLLECTION   #
    #########################
    # save images required for the calibration
    while save_img:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Store color and depth in frames to align
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame =  frames.get_depth_frame()
        aligned_depth_data =  np.asanyarray(aligned_depth_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_data, alpha=0.03), cv2.COLORMAP_JET)
        color_frame = np.asanyarray(color_frame.get_data())

        # Stack both images horizontally
        images = np.hstack((color_frame, depth_colormap))
        cv2.putText(images,"Press Y to record image number "+str(count),
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # if key is pressed save imgage
        keypress = cv2.waitKey(1)
        if ord('y') == keypress:
            # save the color image
            cv2.imwrite(join(img_path,'calibration_'+str(count))+".jpg",color_frame)
            # save the distances from the depth camera
            distances = []
            width,height,_ = color_frame.shape
            for row in range(width):
                dist_row = []
                for col in range(height):
                    dist_row.append(aligned_depth_frame.get_distance(row,col))
                distances.append(dist_row)
            distances = np.array(distances)
            np.save(join(img_path,'calibration_dist_'+str(count)), distances)
            count +=1
        elif ord('q') == keypress:
            break
finally:

    # Stop streaming
    pipeline.stop()


#########################
#     CALIBRATION       #
#########################
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
x_grid = 9
y_grid = 6
images = glob(join(img_path,'*.jpg'))
depth = glob(join(img_path,'*.npy'))
images.sort()
depth.sort()
objpoints = [] # 3d point in real world space
imgpoints = [] # 3d image points.

for iname, dname in zip(images, depth):
    # Arrays to store object points and image points from all the images.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # We start with the bottom floor
    objp = np.zeros((x_grid*y_grid,3), np.float32)
    # Divide all the points by 100 so they are in meters
    objp[:,:2] = ((np.mgrid[0:x_grid,0:y_grid]*2).T.reshape(-1,2))/100
    # add the height to the second image
    if "_1" in iname:
        objp[:,2] = np.ones((x_grid*y_grid))*-0.0334
    # get the color image
    img = cv2.imread(iname)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # get the depth image
    depth = np.load(dname)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x_grid,y_grid),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        list_corners = corners2.tolist()
        corners_3d = []
        for i in range(len(list_corners)):
            # for each pixel value find the z value and add it
            x,y= list_corners[i][0]
            z_dist = depth[int(y),int(x)]
            corners_3d.append([list_corners[i][0]+[z_dist]])
        imgpoints.append(np.array(corners_3d))

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (x_grid,y_grid), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)

# create the homography
imgpoints = np.array(imgpoints,dtype='float32').reshape(-1,1,3)
objpoints= np.array(imgpoints,dtype='float32').reshape(-1,1,3)
H, mask = cv2.findHomography(imgpoints, objpoints, cv2.RANSAC)

# Determine error
error = []
for source, dest in zip(imgpoints,objpoints):
    estimated_dest = np.dot(H,source[0])
    error.append(np.linalg.norm(dest-estimated_dest))
print("AVERAGE EUCLIDEAN ERROR IN METERS:", np.mean(error))
