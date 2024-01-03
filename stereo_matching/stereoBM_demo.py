

import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt
import os
import json
import sys 
 
# Stereo matching example from https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

# Load the yaml with the common options
options = yaml.safe_load(open('../configs/common.yaml', 'r'))
camera_baseline = options['camera_baseline']*100 # 0.25m converted to millimeters
camera_focal_length = options['camera_focal_length'] # 32 millimeters


imgs_dir = '../data/example_pair/'

view_id = 0

img0_path = imgs_dir + "view_"+ "%04i"%view_id +"_left.png"
img1_path = imgs_dir + "view_"+ "%04i"%view_id +"_right.png"

save_path = imgs_dir + "BlockMatching_demo/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

savename_disp = save_path + "view_"+ "%04i"%view_id + "_disparity_norm.png"
savename_depth = save_path + "view_"+ "%04i"%view_id + "_depth_norm.png"
savename_params = save_path + "view_"+ "%04i"%view_id + "_BM_params.json"

imgL = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

def nothing(x):
    pass

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,25,nothing)
 
# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()
 
while True:
 
    Left_nice = imgL.copy()
    Right_nice = imgR.copy()
 
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity_vis = (disparity/16.0 - minDisparity)/numDisparities
    # Displaying the disparity map
    cv2.imshow("disp",disparity_vis)


    #disparity = disparity/16.0
    #disparity_tmp = disparity.copy()
    disparity += 1e-4 # to avoid division with zero
    depth = camera_baseline*camera_focal_length / disparity
    #print(np.amax(depth))
    depth_vis = depth / np.amax(depth)
    #cv2.imshow("depth", depth)
    #print(depth)


    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break

#print(np.unique(disparity_vis))
#print(np.unique(depth_vis))
#print(np.unique(disparity))
#print(np.unique(depth))
#print(np.unique(disparity_tmp))

cv2.imwrite(savename_disp, disparity_vis*256.0)
cv2.imwrite(savename_depth, depth_vis*256.0)
# save the stereoBM values used
BM_params = {'numDisparities': numDisparities, 'blockSize': blockSize, 'preFilterType': preFilterType,
             'preFilterSize': preFilterSize, 'preFilterCap':preFilterCap, 'textureThreshold':textureThreshold, 
             'uniquenessRatio':uniquenessRatio, 'speckleRange':speckleRange, 'speckleWindowSize':speckleWindowSize, 
             'disp12MaxDiff':disp12MaxDiff, 'minDisparity':minDisparity}
with open(savename_params, "w") as outfile:
    json.dump(BM_params, outfile, indent=4)
