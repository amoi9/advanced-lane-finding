import numpy as np
from lane_pixel_finder import find_lane_pixels

'''
Calculates the curvature of polynomial functions in meters.
'''
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def measure_curvature_and_offset_real(binary_warped, direction='left'):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
    if direction == 'left':
        x = leftx
        y = lefty
    else:
        x = rightx
        y = righty
    curature = measure_curvature_real_with_pixels(binary_warped.shape[1], binary_warped.shape[0], x, y)
    offset = measure_offset_real(binary_warped.shape[1], x)
    return curature, offset

def measure_curvature_real_with_pixels(binary_warped_width, binary_warped_length, x, y):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped_length-1, binary_warped_length)
    
    # Fit a second order polynomial to each using `np.polyfit`
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### calculation of R_curve (radius of curvature) #####
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

def measure_offset_real(binary_warped_width, x):
    return (x[0] - binary_warped_width / 2) * xm_per_pix