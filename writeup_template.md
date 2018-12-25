

[//]: # (Image References)

[image1]: ./output_images/undistorted_vs_original.png "Undistorted vs original"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/color_gradient_threshed.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/poly_fit_lanes.png "Fit Visual"
[image6]: ./output_images/marked_output.png "Output"
[video1]: ./project_video.mp4 "Video"
[undistorted]: ./output_images/test1_undist.png "Undistorted test image"
  

## Camera Calibration

The code for this step is contained in the file called `code/undistort.py`. 

In the first code cell of `advanced.ipynb` I invoked method `prepare_obj_img_points` from the imported file. This 
prepares "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here 
I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each 
calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy 
of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the 
(x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

In the third code cell of `advanced.ipynb` I invoked the `cal_undistort` method. This used the output `objpoints` and 
`imgpoints` to compute the camera calibration and distortion coefficients using 
the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` 
function and obtained this result: 

![alt text][image1]

## Pipeline (single images)

### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

In the second code cell in `advanced.ipynb` I invoked the `cal_undistort` method from `code/undistort.py` to get a 
distortion-corrected image:
![alt text][undistorted]

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 5 through 
46 in `code/color_and_gradient_threshed.py`). Code to generate the output is in the 4th cell of `advanced.ipynb`. Here's an 
example of my output for this step.
![alt text][image3]

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform into birds-view includes a function called `warper()`, which appears in lines 4 through 10 in the 
file `code/perspective_transform.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) 
and destination (`dst`) points. 

I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 45, img_size[1] / 2 + 100],
    [((img_size[0] / 6)), img_size[1]],
    [(img_size[0] * 5 / 6) + 130, img_size[1]],
    [(img_size[0] / 2 + 95), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 460      | 320, 0        | 
| 213.3, 720      | 320, 720      |
| 1196.7, 720     | 960, 720      |
| 735, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image 
and its warped counterpart to verify that the lines appear parallel in the warped image. (Code to generate image in the 5th code cell)

![alt text][image4]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the 6th and 8th code cell of the notebook for a test image (`test_images/test3.jpg`), I run the steps:
  1. Undistort the image
  2. Run color and gradients transforms to get a binary image
  3. Apply perspective transform on the binary image to get a warped output
  4. Find lane pixels from the binary warp image (using the sliding window appraoch, the `find_lane_pixels` method from 
  `code/lane_pixel_finder.py`)
  5. Fit the lane lines with a 2nd order polynomial (using the `fit_polynomial` method from `code/fit_polynomial.py`)
  
The steps of the sliding window appraoch to detect the lane pixels and prepare the pixels to fit a polynomial are as the following:
1. Take a histogram of the bottom half of the image
2. Find the peak of the left and right halves of the histogram, which will be the starting point for the left and right lines
3. Set up paramters of the slidng windows, e.g. number of windows, width of the window margin, minimum number of pixels found
in the window, height of the window, .
4. Loop through each window, and do this: 
   1. find the boundaries of our current window
   2. identify the nonzero pixels in x and y within the window
   3. append these indices to the lists
   4. if found > minpix pixels from the previous step, recenter next window based on the mean position of these pixels
5. Concatenate the arrays of indices
6. Extract left and right line pixel positions

Then use `np.polyfit` to fit polynomials, and generate x and y values for plotting.
 
The output is like this:

![alt text][image5]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 8 through 32 in `code/curvature_and_offset.py`. 

`measure_curvature_real_with_pixels` is for the curvature calculation, I fit a second order polynomial of the detected
lane using `np.polyfit`, then use the "Radius of Curvature" formula from the the "Measuring Curvature I" lesson.

For the offset calculation I did the following:
* Assume that the vehicle's position is in the middle of the image.
* Find the center of the lane from the fitted polynomials.
* Find the difference between vehicle position and lane center.

I used the polynomial output from the curvature calculation, which already counted the conversion from pixels to meters.
So in the code below I only convert when needed but not everywhere: 
```
def measure_offset_real(img_shape, left_fit, right_fit):
    y = ym_per_pix * img_shape[0]
    l_fitValue = left_fit[0]* y**2 + left_fit[1]*y + left_fit[2]
    r_fit_Value = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    lane_center_pos = (l_fitValue + r_fit_Value) /2
    
    return lane_center_pos - img_shape[1] / 2 * xm_per_pix
``` 


### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the 10th code cell of `advanced.ipynb`. 
Here is an example of my result on a test image:

![alt text][image6]

---

## Pipeline (video)

Here's a [link to my video result](./project_video_output.mp4)

---

## Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and 
how I might improve it if I were going to pursue this project further.  

### 1. Approach

The code is in `code/pipeline.py`. I got there by debugging with single images from the video clips and ran steps 
mentioned previously in the doc.

The `init_process` method does the same as previously decribed for a single image, it 
is used for the first frame processsing, as well as when we fail sanity checks multiple frames in row and need to restart.
This method finds lane pixels starting from the peak of the histogram, iterating the sliding windows, and extracting the pixels with 
non-zero x and y values.

The `search_around_previous` method takes advantage of previously calculated polynomial coefficients, and only search around
in a margin within the previous line position. To achieve that I instantiated a `left_line` and `right_line` which are instances 
of the `Line` class, and records the recent processed frame for the respective lanes. I only record those passed the sanity checks,
namely lanes have similar curvature, are separated by approximately the right distance horizontally and roughly in parallel.

I noticed the sanity checks improved the 28~30th second of the project vedio when a car from the right passed by.

In order to know when to fall back to `init_process`, I keep track of the `current_frame` number, also record the good frame 
numbers in the `Line` instances.

### 2. Potential improvements

I wanted to smooth the drawing by averaging out the recent past measurements, but either I didn't manage to find the 
right `numpy` methods, or naive averaging doesn't work. I wasn't able to get the drawing working with averaged results.
I could spent more time on that.

My pipeline doesn't handle the first frame failure case well, which can be improved.

I tried the pipeline in the `challenge_video.mp4`, it seems not working well when there are marks in between the lanes 
and shadows under differnt lighting. I could sanity check that the lane positions don't move too much horizontally, e.g.
setting a threshold on the `offset`.

The thresholds for sanity checks aren't tested across videos.
