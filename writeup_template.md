

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

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

In the second code cell in `advanced.ipynb` I invoked the `cal_undistort` method from `code/undistort.py` to get a 
distortion-corrected image:
![alt text][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 5 through 
46 in `code/color_and_gradient_threshed.py`). Code to generate the output is in the 4th cell of `advanced.ipynb`. Here's an 
example of my output for this step.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 4 through 10 in the 
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

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the 6th and 8th code cell of the notebook for a test image (`test_images/test3.jpg`), I run the steps:
  1. Undistort the image
  2. Run color and gradients transforms to get a binary image
  3. Apply perspective transform on the binary image to get a warped output
  4. Find lane pixels from the binary warp image (using the `find_lane_pixels` method from `code/lane_pixel_finder.py`)
  5. Fit the lane lines with a 2nd order polynomial (using the `fit_polynomial` method from `code/fit_polynomial.py`)
 
The output is like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 8 through 42 in `code/curvature_and_offset.py`. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the 10th code cell of `advanced.ipynb`. 
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
