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
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted
    # cv2.imshow('un')


### STEP 3 functions
# 3. Use color transforms, gradients, etc., to create a thresholded binary image.

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    print(['thresh0',thresh[0]])
    # print(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # print(grad_binary)
    return grad_binary

def mag_sobel_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
   # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def dir_sobel_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def bgr_threshold(img, color = 'r', thresh = (0,255)):
    # img is BGR format
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    binary = np.zeros_like(B)
    
    if color == 'r':
        binary[(R >= thresh[0]) & (R <= thresh[1])] = 1
    elif color == 'g':
        binary[(G >= thresh[0]) & (G <= thresh[1])] = 1
    elif color == 'b':
        binary[(B >= thresh[0]) & (B <= thresh[1])] = 1
    else:
        print('Please choose color from r, g, b !')
    return binary

def hls_threshold(img, color = 'h', thresh = (0,255)):
    #ing is BGR format
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    binary = np.zeros_like(H)
    if color == 'h':
        binary[(H > thresh[0]) & (H <= thresh[1])] = 1
    elif color == 'l':
        binary[(L > thresh[0]) & (L <= thresh[1])] = 1
    elif color == 's':
        binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    else:
        print('Please choose color from h, l, s !')
    return binary
    
        
### STEP 4 functions
# get unwarp matrix from test image
def unwarp_cal(mtx,dist):
    test_img_path = 'test_images/straight_lines1.jpg'
    test_img = cv2.imread(test_img_path)
    test_img = undistort(test_img,mtx, dist)
    img_size = (test_img.shape[1], test_img.shape[0])
    offsetX = 200;
    offsetY = 0;
    # src pts n
    src = np.float32([(566, 470), (724, 470), (1040, 676), (269, 676)])
    dst = np.float32([[offsetX, offsetY], [img_size[0]-offsetX, offsetY], 
                                        [img_size[0]-offsetX, img_size[1]-offsetY], 
                                        [offsetX, img_size[1]-offsetY]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M

def warpPerspective(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

# Step 5 functions


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped):
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

def get_lane_curvature(lane_img):
    binary_curv_img = lane_img[:,:,1]
    xmax = binary_curv_img.shape[1]
    ymax = binary_curv_img.shape[0]
    pix_coor = np.argwhere(binary_curv_img > 100)
    leftx = np.array([])
    lefty = np.array([])
    rightx = np.array([])
    righty = np.array([])
    for xy in pix_coor:
        # print(xy)
        if xy[1] <= xmax/2 :
            leftx = np.append(leftx,xy[1])
            lefty = np.append(lefty,xy[0])
        else:
            rightx = np.append(rightx,xy[1])
            righty = np.append(righty,xy[0])

    # Reverse to match top-to-bottom in y
    # lefty = lefty[::-1]
    # righty = righty[::-1]  

    # cruve fit

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty  = np.linspace(0, binary_curv_img.shape[0]-1, binary_curv_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    plt.plot(leftx,lefty, 'o', color='red', markersize=1)
    plt.plot(rightx,righty, 'o', color='blue', markersize=1)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() 
    plt.show()

    # calculate curvature in pixel
    left_y_eval = np.max(ploty)
    right_y_eval = np.max(ploty)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*left_y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*right_y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)

    # calculate curvature in meter
    # Fit new polynomials to x,y in world space
    ym_per_pix = (30*(700/880)) /720
    xm_per_pix = 3.7/880
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*left_y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*right_y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    return left_fitx,right_fitx,ploty


def draw_lane(image, undist,warped,M_unwarp,left_fitx, right_fitx,ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    plt.imshow(color_warp)
    plt.show()
    ret,Invert_M_unwarp = cv2.invert(M_unwarp)
    print(Invert_M_unwarp)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Invert_M_unwarp, (image.shape[1], image.shape[0])) 
    plt.imshow(newwarp)
    plt.show()
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
 
    return result

def calculate_offset():
    return

def find_lane_pipeline(image):
    # undistort
    mtx, dist, rvecs, tvecs = load_cam_cal('cam_cal')

    undist = cv2.undistort(image, mtx, dist, None, mtx)

    binary_abs = abs_sobel_thresh(test_img, orient='x', thresh=(20,150))

    binary_hls = hls_threshold(test_img,  color = 's', thresh=(170,255))

    warped = warpPerspective(combined_binary, M_unwarp)

    result = image
    return result

def load_video():
    video_path = 'project_video.mp4'
    # clip1 = VideoFileClip(video_path))
    clip1 = VideoFileClip(video_path).subclip(0,5)
    white_clip = clip1.fl_image(process_image_function)
