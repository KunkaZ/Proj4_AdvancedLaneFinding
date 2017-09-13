from lane_finding import *
mtx, dist, rvecs, tvecs = load_cam_cal('cam_cal')
M_unwarp = unwarp_cal(mtx, dist)
left_cur_k      = 0
left_cur_k_pre  = 0
right_cur_k     = 0
right_cur_k_pre  = 0
frame_index = 0
def process_image(image_rgb):
    global frame_index
    # return image_rgb
    # image_rgb is from video frame, all function in lane_finding is using bgr image
    image_bgr = image_rgb[...,::-1]
    global mtx, dist, rvecs, tvecs, M_unwarp
    # Step 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

    # Step 2: Apply a distortion correction to raw images.
    test_img = undistort(image_bgr,mtx, dist)
    # Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
    binary_hls = np.zeros_like(test_img[:,:,1])
    binary_abs = np.zeros_like(test_img[:,:,1])
    binary_abs = abs_sobel_thresh(test_img, orient='x', thresh=(20,150))
    # binary_hls = hls_threshold(test_img,  color = 's', thresh=(170,255))  #170,255
    binary_yellow   = select_yellow(test_img)
    binary_white    = select_white(test_img)
    binary_hsv      = hsv_combine(test_img)
    
    combined_binary2 = np.zeros_like(binary_abs)  # use abs and hls
    combined_binary2[((binary_abs == 1) | (binary_hls == 1)) | (binary_yellow == 1) | (binary_white == 1)] = 1

    # Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
    # top_left, top_right, bottom_right, bottom_left. 
    warped = warpPerspective(combined_binary2, M_unwarp)

    # Step 5: Detect lane pixels and fit to find the lane boundary
    window_centroids = find_window_centroids(warped)




    # If we found any window centers
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    if len(window_centroids) > 0:
        #calculate center shift
        


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

    binary_curv_img = output[:,2,:]

    # 6. Determine the curvature of the lane and vehicle position with respect to center.
    left_fitx,right_fitx,ploty,left_curverad,right_curverad,center_shift = get_lane_curvature(output)
    # """
    # print('center_shift_mine',center_shift)
    # 7. plot lane on the image
    test_img_rgb = image_rgb
    result = draw_lane(test_img_rgb,test_img,warped,M_unwarp,left_fitx, right_fitx,ploty)

    global left_cur_k,left_cur_k_pre,right_cur_k,right_cur_k_pre
    if left_cur_k == 0:
        left_cur_k = left_cur_k_pre = left_curverad
        right_cur_k = right_cur_k_pre = right_curverad
    if left_curverad > 10*left_cur_k_pre:
        left_curverad = left_cur_k_pre
    left_cur_k = 0.9*left_cur_k_pre + 0.1*left_curverad
    left_cur_k_pre = left_cur_k
    if right_curverad > 10*left_cur_k_pre:
        left_curverad = left_cur_k_pre
    right_cur_k = 0.9*right_cur_k_pre + 0.1*right_curverad
    right_cur_k_pre = right_cur_k

    # overwrite center shift with new method
    camera_position = image_rgb.shape[1]/2
    xm_per_pix = 3.7/880
    center_shift = ((left_fitx[-1] + right_fitx[-1])/2 - camera_position)*xm_per_pix
    # print('center_shift',center_shift)


    avg_cur = (left_cur_k+right_cur_k)/2
    if center_shift > 0.0:
        center_shift_text = 'Vehicle is ' + str(round(center_shift,2)) +' [m] left of center.'
    elif center_shift < 0.0:
        center_shift_text = 'Vehicle is ' + str(round(-center_shift,2)) +' [m] right of center.'
    else:
        center_shift_text = 'Vehicle is at center.'
    text = 'Radius of curvature: '+str(int(avg_cur)) +' [m] '


    font = cv2.FONT_HERSHEY_SIMPLEX

    result = cv2.putText(result,text,(40,100), font, 1, (255,255,255), 2, cv2.LINE_AA)

    result = cv2.putText(result,center_shift_text,(40,150), font, 1, (255,255,255), 2, cv2.LINE_AA)

    if debug_switch:
        # print('yellow shape:',binary_yellow.shape)
        # print('white shape:',binary_white.shape)
        f, axarr2 = plt.subplots(2,2)
        f.tight_layout()
        axarr2[0,0].imshow(image_rgb, cmap='gray')
        # axarr2[0,0].imshow(test_img[:,:,::-1], cmap='gray')
        axarr2[0,0].set_title('original', fontsize=10)
        axarr2[0,1].imshow(combined_binary2, cmap='gray')
        axarr2[0,1].set_title('combined_binary2', fontsize=10)
        axarr2[1,0].imshow(warped, cmap='gray')
        axarr2[1,0].set_title('warped', fontsize=10)
        axarr2[1,1].imshow(result, cmap='gray')
        axarr2[1,1].set_title('final result', fontsize=10)
        plt.show()
        # './debugimage/' + 
        path = str(frame_index)+'.png'
        path_result = str(frame_index)+'result.png'
        cv2.imwrite(path,image_bgr)
        cv2.imwrite(path_result,result)
        frame_index +=1
    return result

# white_output = 'test_videos_output/solidWhiteRight.mp4'
video_path = 'project_video.mp4'
output_video = 'output_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds

# clip1 = VideoFileClip(video_path).subclip(0.5,1)
clip1 = VideoFileClip(video_path)

# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output_video, audio=False)