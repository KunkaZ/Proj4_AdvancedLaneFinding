## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./report_image/orig.png "Original"
[image1]: ./report_image/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./report_image/binary_combo_example.png "Binary Example"
[image4]: ./report_image/warp_verify.png "Warp Example"
[color_fit_lines]: ./report_image/color_fit_lines.png "Fit Visual"
[image6]: ./report_image/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"
[warped_lane]: ./report_image/warped_binary.png "warped_lane"
[curvature_eq]: ./report_image/curvature_eq.png "curvature_eq"
[after1]: ./report_image/discussion/after.png "after1"
[after2]: ./report_image/discussion/after2.png "after2"
[before1]: ./report_image/discussion/before1.png "before1"
[before2]: ./report_image/discussion/before2.png "before2"
[binary]: ./report_image/discussion/binary.png "binary"


---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in function `camral_cal()` in `lane_finding.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:   
Undistorted image
![alt text][image1]


### Pipeline (single images)


#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Cameral calibration matrix and undistortion matrix are used for undistort this image. It's done in function `undistort()` in `lane_finding.py`
![alt text][image0]
Undistorted image:   
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


Different thresholding methods were implemented in `lane_finding.py`. 

| threshold variable        | function name   | 
|:-----------------------------:|:---------------------------:| 
| sobel gradient                | `abs_sobel_thresh()`        | 
| sobel gradient magnitude      | `mag_sobel_thresh()`      |
| sobel gradient direction      | `dir_sobel_threshold()`     |
| RBG color                     | `bgr_threshold()`        |
| HLS color                     | `hls_threshold()`        |
| LAB color                     | `lab_threshold()`        |



After tied different combination for thresholding methold, I used a combination of HLS color and sobel gradient thresholds to generate a binary image.  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

An unwarp calibration is done after camera calibration. It's implemented in function `unwarp_cal` in `lane_finding.py`. It returns a warp transformation matrix.

 Source point are manually chosen from a calibration image and hard codeding in function `unwarp_cal`. Destination points are generated with offset and image size. 

```python
    offsetX = 200;
    offsetY = 0;
    src = np.float32([(564, 470), (720, 470), (1120, 720), (190, 720)])
    dst = np.float32([[offsetX, offsetY], [img_size[0]-offsetX, offsetY], 
                                        [img_size[0]-offsetX, img_size[1]-offsetY], 
                                        [offsetX, img_size[1]-offsetY]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 564, 470      | 320, 0        | 
| 720, 470      | 320, 720      |
| 1120, 720     | 960, 720      |
| 190, 720      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. (Note that this images is not the image for pipeline demonstration. This image has a straight lane and is good for warp calibration.)

![alt text][image4]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


With applying thresholding method mentioned in point 2, I get a binary image of warped lane:
![alt text][warped_lane]
Each lane lines are fit with a 2nd order polynomial, shown in green curve in following image:   
![alt text][color_fit_lines]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Lane curvature and vehicle position is caculated in function `get_lane_curvature()` in `lane_finding.py`.   
Lane curvature was calculated according to a formula for the radius of curvature at any point x for the curve y = f(x)![alt text][curvature_eq ].

Vehicle position shift is ca
x axis values of point in left lane line and right lane line in warped binary image shown in bullet 4 are used for calculation of vehicle position shift. Following equation is used   
### center_x_meter = ( (left_lane_x + right_lane_x) - image_size_x_max ) /2 * meter_per_pix_x_axis   
Code implementation is in`get_lane_curvature()` from line 309~320.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `draw_lane()` in `lane_finding.py`.  Here is an example of my result on a test image:

![alt text][image6

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
