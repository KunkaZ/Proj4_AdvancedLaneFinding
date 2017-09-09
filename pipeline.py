from lane_finding import *
# plt.ion()
# import matplotlib as mp
# mp.interactive(True)

# Step 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
mtx, dist, rvecs, tvecs = load_cam_cal('cam_cal')
M_unwarp = unwarp_cal(mtx, dist)

test_image_name = 'test_images/test1.jpg' 


# Step 2: Apply a distortion correction to raw images.
images = glob.glob(test_imges_path+'*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    # cv2.imshow(fname,img)
    # cv2.waitKey(500)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # cv2.imshow(fname+'undist',undist)
    # cv2.waitKey(500)
    cv2.destroyAllWindows()

# Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
# choose three 
test_img_orig = cv2.imread(test_image_name)
print(images[0])
test_img = undistort(test_img_orig,mtx, dist)

if 1: # test each function
    binary_abs = abs_sobel_thresh(test_img, orient='x', thresh=(20,150))
    print('imshow(binary)')


    print('mag_sobel_thresh')
    # binary1 = mag_sobel_thresh(test_img, mag_thresh=(70,150))
    binary_mag = mag_sobel_thresh(test_img, mag_thresh=(50,100))

    print('bgr_threshold')
    binary_bgr = bgr_threshold(test_img,color = 'b', thresh=(80,120))

    print('hls_threshold')
    binary_hls = hls_threshold(test_img,  color = 's', thresh=(170,255))
    binary_hls_h = hls_threshold(test_img,  color = 'h', thresh=(15,100))

    binary_left = dir_sobel_threshold(test_img, sobel_kernel=3, thresh=(np.pi/8, 3*np.pi/8))

    binary_right = dir_sobel_threshold(test_img, sobel_kernel=3, thresh=(5*np.pi/8, 7*np.pi/8))

    # overlap binary image
    binary_img1 = binary_abs
    binary_img2 = binary_mag
    binary_img3 = binary_bgr
    binary_img4 = binary_hls
    binary_img5 = test_img
    binary_img6 = binary_right
    result = np.dstack(( np.zeros_like(binary_img1), binary_img1, binary_img2))

    combined_binary = np.zeros_like(binary_abs)
    combined_binary[((binary_abs == 1) | (binary_hls == 1)) | (binary_mag == 1)] = 1

    combined_binary2 = np.zeros_like(binary_abs)  # use abs and hls
    combined_binary2[(binary_abs == 1) | (binary_hls == 1) ] = 1

    f, axarr = plt.subplots(2,3)
    f.tight_layout()
    axarr[0,0].imshow(binary_img1, cmap='gray')
    axarr[0,0].set_title('abs_sobel_thresh', fontsize=15)

    axarr[0,1].imshow(binary_img2, cmap='gray')
    axarr[0,1].set_title('mag_sobel_thresh', fontsize=15)

    axarr[1,0].imshow(binary_hls_h, cmap='gray')
    axarr[1,0].set_title('binary_hls_h', fontsize=15)

    axarr[1,1].imshow(binary_img4, cmap='gray')
    axarr[1,1].set_title('hls_threshold', fontsize=15)

    axarr[0,2].imshow(combined_binary, cmap='gray')
    axarr[0,2].set_title('combined_binary', fontsize=15)

    axarr[1,2].imshow(combined_binary2, cmap='gray')
    axarr[1,2].set_title('combined_binary2', fontsize=15)

    plt.show()

# Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
# top_left, top_right, bottom_right, bottom_left. 
warped = warpPerspective(combined_binary, M_unwarp)

plt.imshow(warped)
plt.show()



# Step 5: Detect lane pixels and fit to find the lane boundary
# Read in a thresholded image
# warped = combined_binary
# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching


window_centroids = find_window_centroids(warped)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
	    # Add graphic points from window mask here to total pixels found 
	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

# Display the final results
plt.imshow(output)
plt.title('window fitting results')
plt.show()
# cv2.imwrite('binary_img.png',output)

binary_curv_img = output[:,2,:]
print(output.shape)

# 6. Determine the curvature of the lane and vehicle position with respect to center.
print('Step 6')
left_curverad,right_curverad,ploty = get_lane_curvature(output)
# """

# 7. plot lane on the image
# rgb = bgr[...,::-1]
test_img_rgb = test_img_orig[...,::-1]
result = draw_lane(test_img_rgb,test_img,warped,M_unwarp,left_curverad, right_curverad,ploty)
plt.imshow(result)
plt.show()