#GOAL:  software pipeline to identify the lane boundaries in a video
# Steps:

# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
# 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# 5. Detect lane pixels and fit to find the lane boundary.
# 6. Determine the curvature of the lane and vehicle position with respect to center.
# 7. Warp the detected lane boundaries back onto the original image.
# 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
output_images_path  = 'output_images/'
test_imges_path     = 'test_images/'
examples_path       = 'examples/'
cameral_cal_path    = 'camera_cal/'
# 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
# chessboard size 9*6
def cameral_cal(filename):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp=np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    print('Calibrating camera ...')
    images = glob.glob(cameral_cal_path+'calibration*.jpg')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray,(9,6),None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    print('finish camera calibration, save matrices ...')

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    pickle.dump( dist_pickle, open(filename, "wb" ) )
    return ret, mtx, dist, rvecs, tvecs

# load calibrated pickle file
# Ex. load_cam_cal('myCal.p')
def load_cam_cal(filename):
    try:
        print("Load matrices from saved file.")
        dist_pickle = pickle.load(open(filename, 'rb'))
        mtx = dist_pickle["mtx"] 
        dist = dist_pickle["dist"] 
        rvecs = dist_pickle["rvecs"] 
        tvecs = dist_pickle["tvecs"] 
    except IOError:
        print("Camera not calibrated.")
        ret, mtx, dist, rvecs, tvecs = cameral_cal(filename)
    return mtx, dist, rvecs, tvecs
        

# distortion correction to a img
def undistort(img,mtx,dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted
    # cv2.imshow('un')

# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
def generate_binary_img():
    return


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    grad_binary = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
   # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary






mtx, dist, rvecs, tvecs = load_cam_cal('cam_cal')

test_image_name = 'test_images/test1.jpg' 


# Step 2: Apply a distortion correction to raw images.
images = glob.glob(test_imges_path+'*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    cv2.imshow(fname,img)
    cv2.waitKey(500)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow(fname+'undist',undist)
    cv2.waitKey(500)
# TODO:

# Step 3: