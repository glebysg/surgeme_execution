## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
record = True
color_record_path = 'data/videos/tool_2.avi'
depth_record_path = 'data/videos/tool22.avi'

color_out = cv2.VideoWriter(color_record_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480)) if record else None
depth_out = cv2.VideoWriter(depth_record_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480)) if record else None

count = 0
try:
    while True:
        # check if the cap is open
        # Wait for a coherent pair of frames: depth and color
        frameset = pipeline.wait_for_frames()
        # Store color and depth in frameset to align
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        # Skip first 5 frames to give Auto exposure time to adjust
        if not color_frame or not depth_frame or count < 5 :
            count += 1
            continue
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame =  np.asanyarray(frameset.get_depth_frame().get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        # reduce brightness
        depth_grayscale =cv2.convertScaleAbs(aligned_depth_frame, alpha=0.05)
        depth_colormap = cv2.applyColorMap(depth_grayscale, cv2.COLORMAP_JET)
        depth_grayscale = cv2.cvtColor(depth_grayscale, cv2.COLOR_GRAY2RGB)
        color_frame = np.asanyarray(color_frame.get_data())
        print("DISTANCE:",frameset.get_depth_frame().get_distance(300,200))

        # Stack both images horizontally
        images = np.hstack((color_frame, depth_grayscale))
        if record:
            color_out.write(color_frame)
            depth_out.write(depth_grayscale)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    cv2.destroyAllWindows()
    pipeline.stop()
    if record:
        color_out.release()
        depth_out.release()
