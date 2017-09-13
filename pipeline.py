from lane_finding import *
# plt.ion()
# import matplotlib as mp
# mp.interactive(True)
global debug_switch
debug_switch = 1
# Step 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
mtx, dist, rvecs, tvecs = load_cam_cal('cam_cal')
M_unwarp = unwarp_cal(mtx, dist)

# test_image_name = 'test_images/test2.jpg' 
# test_image_name = 'test_images/straight_lines1.jpg'
test_image_name = '5.png' 

# Step 2: Apply a distortion correction to raw images.
images = glob.glob(test_imges_path+'*.jpg')
# for idx, fname in enumerate(images):
#     img = cv2.imread(fname)
#     # cv2.imshow(fname,img)
#     # cv2.waitKey(500)
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     # cv2.imshow(fname+'undist',undist)
#     # cv2.waitKey(500)
#     cv2.destroyAllWindows()

# Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
# choose three 
test_img_orig = cv2.imread(test_image_name)

plt.imshow(test_img_orig[:,:,::-1])
plt.show()

print(images[0])
test_img = undistort(test_img_orig,mtx, dist)

plt.imshow(test_img[:,:,::-1])
plt.show()

img_size = (test_img.shape[1], test_img.shape[0])
offsetX = 200;
offsetY = 0;
# src pts n
src = np.float32([(564, 470), (720, 470), (1120, 720), (190, 720)])
dst = np.float32([[offsetX, offsetY], [img_size[0]-offsetX, offsetY], 
                                    [img_size[0]-offsetX, img_size[1]-offsetY], 
                                    [offsetX, img_size[1]-offsetY]])

test_img_warped = warpPerspective(test_img, M_unwarp)
color = [0,0,255]
thickness = 5
print('src[1]')
print(src[1])

# f, axarr = plt.subplots(2,4)
# f.tight_layout()
if 0:
    f,axeplt = plt.subplots(1,2)
    f.tight_layout()
    cv2.line(test_img, tuple(src[0]),tuple(src[1]), color, thickness)   
    cv2.line(test_img, tuple(src[1]),tuple(src[2]), color, thickness)    
    cv2.line(test_img, tuple(src[2]),tuple(src[3]), color, thickness)   
    cv2.line(test_img, tuple(src[3]),tuple(src[0]), color, thickness)   
    # plt.imshow(test_img[:,:,::-1])
    test_img_rgb =test_img[:,:,::-1]

    axeplt[0].imshow(test_img_rgb)
    axeplt[0].set_title('Undistorted Image with source points drawn', fontsize=10)

    cv2.line(test_img_warped, tuple(dst[0]),tuple(dst[1]), color, thickness)   
    cv2.line(test_img_warped, tuple(dst[1]),tuple(dst[2]), color, thickness)    
    cv2.line(test_img_warped, tuple(dst[2]),tuple(dst[3]), color, thickness)   
    cv2.line(test_img_warped, tuple(dst[3]),tuple(dst[0]), color, thickness)   

    axeplt[1].imshow(test_img_warped[:,:,::-1])
    axeplt[1].set_title('Warped result with dest. points drawn', fontsize=10)

    plt.show()

# exit()
# cv2.line(img, (leftLine[0],leftLine[1]), (leftLine[2],leftLine[3]), color, thickness)    
# cv2.line(img, (rightLine[0],rightLine[1]), (rightLine[2],rightLine[3]), color, thickness)




binary_abs = abs_sobel_thresh(test_img, orient='x', thresh=(20,110))
# print('imshow(binary)')


# print('mag_sobel_thresh')
# binary1 = mag_sobel_thresh(test_img, mag_thresh=(70,150))
binary_mag = mag_sobel_thresh(test_img, mag_thresh=(50,100))

# print('bgr_threshold')
binary_bgr = bgr_threshold(test_img,color = 'b', thresh=(80,120))

# print('hls_threshold')
binary_hls = hls_threshold(test_img,  color = 's', thresh=(170,255))
binary_hls_h = hls_threshold(test_img,  color = 'h', thresh=(15,100))

binary_lab_b = lab_threshold(test_img,  color = 'b', thresh=(150,200))

binary_left = dir_sobel_threshold(test_img, sobel_kernel=3, thresh=(np.pi/8, 3*np.pi/8))

binary_right = dir_sobel_threshold(test_img, sobel_kernel=3, thresh=(5*np.pi/8, 7*np.pi/8))

binary_yellow   = select_yellow(test_img)
binary_white    = select_white(test_img)
binary_hsv      = hsv_combine(test_img)
# overlap binary image

combined_binary = np.zeros_like(binary_yellow)
# combined_binary[((binary_abs == 1) | (binary_hls == 1)) | (binary_yellow == 1) | (binary_white == 1)] = 1
combined_binary[(binary_yellow == 1) | (binary_white == 1)] = 1

combined_binary2 = np.zeros_like(binary_abs)  # use abs and hls
combined_binary2[(binary_abs == 1) | (binary_hls == 1) ] = 1
# combined_binary2[(binary_yellow == 1) | (binary_white == 1) ] = 1
# plt.imshow(combined_binary2, cmap='gray')
# plt.show()

# Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
# top_left, top_right, bottom_right, bottom_left. 
warped = warpPerspective(combined_binary2, M_unwarp)

# plt.imshow(warped, cmap='gray')
# plt.show()
# exit()
print('yellow shape:',binary_yellow.shape)
print('white shape:',binary_white.shape)
f, axarr2 = plt.subplots(2,2)
f.tight_layout()
print('binary_hsv',binary_hsv.shape)
axarr2[0,0].imshow(binary_hsv, cmap='gray')
# axarr2[0,0].imshow(test_img[:,:,::-1], cmap='gray')
axarr2[0,0].set_title('binary_hsv', fontsize=10)


axarr2[0,1].imshow(binary_yellow, cmap='gray')
axarr2[0,1].set_title('yellow', fontsize=10)

axarr2[1,0].imshow(combined_binary, cmap='gray')
axarr2[1,0].set_title('combined_binary', fontsize=10)

axarr2[1,1].imshow(test_img, cmap='gray')
axarr2[1,1].set_title('combined_binary2', fontsize=10)
plt.show()


# plt.show()
# exit()

# Step 5: Detect lane pixels and fit to find the lane boundary
# Read in a thresholded image
# warped = combined_binary
# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

print('call find_window_centroids---------')
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
        # print('x')
        # print(window_centroids[level][1])
        # print(level)

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

f, axarr = plt.subplots(2,2)
f.tight_layout()
axarr[0,0].imshow(warped, cmap='gray')
axarr[0,0].set_title('warped', fontsize=10)

axarr[0,1].imshow(output, cmap='gray')
axarr[0,1].set_title('output', fontsize=10)

axarr[1,0].imshow(l_points, cmap='gray')
axarr[1,0].set_title('l_points', fontsize=10)

axarr[1,1].imshow(r_points, cmap='gray')
axarr[1,1].set_title('r_points', fontsize=10)

binary_curv_img = output[:,2,:]
print(output.shape)
# exit()
# 6. Determine the curvature of the lane and vehicle position with respect to center.
print('Step 6')
left_fitx,right_fitx,ploty,left_curverad,right_curverad,center_shift = get_lane_curvature(output)
    # """
# """
# exit()
# 7. plot lane on the image
# rgb = bgr[...,::-1]
test_img_rgb = test_img_orig[...,::-1]
result = draw_lane(test_img_orig,test_img,warped,M_unwarp,left_fitx, right_fitx,ploty)
font = cv2.FONT_HERSHEY_SIMPLEX
# center_shift =str(round(center_shift,2))
if center_shift > 0.0:
    center_shift_text = 'Vehicle is ' + str(round(center_shift,2)) +' [m] left of center.'
elif center_shift < 0.0:
    center_shift_text = 'Vehicle is ' + str(round(-center_shift,2)) +' [m] right of center.'
else:
    center_shift_text = 'Vehicle is at center.'
avg_cur = (left_curverad + right_curverad)/2
text = 'Radius of curvature: '+str(int(avg_cur)) +' [m] '
result = cv2.putText(result,text,(40,100), font, 1, (255,255,255), 2, cv2.LINE_AA)

result = cv2.putText(result,center_shift_text,(40,150), font, 1, (255,255,255), 2, cv2.LINE_AA)

plt.imshow(result)
plt.show()

# f, axarr = plt.subplots(2,4)
# f.tight_layout()
# axarr[0,0].imshow(binary_img1, cmap='gray')
# axarr[0,0].set_title('abs_sobel', fontsize=10)

# axarr[0,1].imshow(binary_img2, cmap='gray')
# axarr[0,1].set_title('mag_sobel', fontsize=10)

# axarr[1,0].imshow(binary_lab_b, cmap='gray')
# axarr[1,0].set_title('lab_b', fontsize=10)

# axarr[1,1].imshow(binary_img4, cmap='gray')
# axarr[1,1].set_title('hls_s', fontsize=10)

# axarr[0,2].imshow(combined_binary, cmap='gray')
# axarr[0,2].set_title('com_binary', fontsize=10)

# axarr[1,2].imshow(combined_binary2, cmap='gray')
# axarr[1,2].set_title('com_binary2', fontsize=10)

# axarr[0,3].imshow(output, cmap='gray')
# axarr[0,3].set_title('wind_fit',fontsize=10)

# axarr[1,3].imshow(result, cmap='gray')
# axarr[1,3].set_title('final')

# plt.show()
