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

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
count = 0
img_path ='calibration'
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
        aligned_depth_frame =  np.asanyarray(frames.get_depth_frame().get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_frame, alpha=0.03), cv2.COLORMAP_JET)
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
            cv2.imwrite(join(img_path,'calibration_'+str(count))+".jpg",color_frame)
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

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = (np.mgrid[0:9,0:6]*2).T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob(join(img_path,'*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img', gray)
    # cv2.waitKey(0)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

######## get the errors ##############3
mean_error = 0
# mean_error2 = 0

for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    # new_points = []
    # for j in range(len(objpoints[i])):
        # point = np.array(objpoints[i][j])
        # point = np.add(point,tvecs[i][0]).reshape(3,1)
        # rot, _ = (cv2.Rodrigues(rvecs[i]))
        # point = np.dot(rot, point)
        # point = np.dot(mtx,point)
        # print(point)
        # new_points.append(point)
    # new_points = np.array(new_points)
    # print(imgpoints.shape)
    # print(new_points.shape)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints)
    # error2 = cv2.norm(imgpoints[i],new_points, cv2.NORM_L2)/len(new_points)
mean_error += error
# mean_error2 += error2

print "total error: ", mean_error/len(objpoints)
# print "total error: ", mean_error2/len(objpoints)
