import cv2
import numpy as np
from line import Line
from undistort import cal_undistort, prepare_obj_img_points
from lane_pixel_finder import find_lane_pixels
from fit_polynomial import fit_polynomial
from curvature_and_offset import measure_curvature_real_with_pixels, measure_offset_real
from color_and_gradient_threshed import threshed_binary_pipeline
from perspective_transform import warper

def init_lines():
    global left_line, right_line, current_frame, objpoints, imgpoints
    left_line = Line()
    right_line = Line() 
    current_frame = 0
    objpoints, imgpoints = prepare_obj_img_points()

def draw_lane(binary_warped, img, undistorted, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    img_size = (undistorted.shape[1], undistorted.shape[0])
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
    
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    return result

def undistorted_and_binary_warped(img):
    undistorted = cal_undistort(img, objpoints, imgpoints)
    binary_img = threshed_binary_pipeline(undistorted, s_thresh=(100, 255), sx_thresh=(30, 100))
    img_size = (undistorted.shape[1], undistorted.shape[0])
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
    binary_warped = warper(binary_img, src, dst)
    return undistorted, binary_warped

curverad_diff_threshold = 800
lane_distance_threshold = 700
coefficient_2d_diff_threshold = 0.0001
def sanity_check_and_set_params(img_shape, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty, left_fit, right_fit):
    left_curverad = measure_curvature_real_with_pixels(img_shape[1], img_shape[0], leftx, lefty)
    left_offset = measure_offset_real(img_shape[1], leftx)
    
    right_curverad = measure_curvature_real_with_pixels(img_shape[1], img_shape[0], rightx, righty)
    right_offset = measure_offset_real(img_shape[1], rightx)
    
    # Check the lanes have similar curvature
    curverad_diff = np.absolute(left_curverad - right_curverad)
    if curverad_diff > curverad_diff_threshold:
        return False
    
    # Check the lanes are separated by approximately the right distance horizontally
    lane_distance = np.absolute(np.average(leftx) - np.average(rightx))
    if lane_distance > lane_distance_threshold:
        return False
    
    # Check the lanes are roughly in parallel
    coefficient_2d_diff = np.absolute(left_fit[0] - right_fit[0])
    if coefficient_2d_diff > coefficient_2d_diff_threshold:
        return False
    
    left_line.set_params(left_fit, left_curverad, left_offset, left_fitx, current_frame)
    right_line.set_params(right_fit, right_curverad, right_offset, right_fitx, current_frame)
    return True

def init_process(img):
    undistorted, binary_warped = undistorted_and_binary_warped(img)
    
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty)
    sanity_check_and_set_params(binary_warped.shape, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty, left_fit, right_fit)
    return draw_lane(binary_warped, img, undistorted, left_fitx, right_fitx, ploty)

def search_around_previous(img):
    undistorted, binary_warped = undistorted_and_binary_warped(img)
    
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Get the previous fit
    prev_left_fit = left_line.current_fit
    prev_right_fit = right_line.current_fit
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty)
    checked = sanity_check_and_set_params(binary_warped.shape, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty, left_fit, right_fit)
    if checked:
        return draw_lane(binary_warped, img, undistorted, left_fitx, right_fitx, ploty)
    return draw_lane(binary_warped, img, undistorted, left_line.allx, right_line.allx, ploty)
    
def process_image(img):
    global current_frame
    # Process the first frame or previous attempts didn't pass sanity-check multiple times in row
    if current_frame == 0 or current_frame - left_line.frame >= 5:
        result = init_process(img)
    else:
        result = search_around_previous(img)
    current_frame += 1
    return result