import numpy as np
from lane_pixel_finder import find_lane_pixels

'''
Calculates the curvature of polynomial functions in meters.
'''
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def measure_curvature_real_with_pixels(img_shape, x, y):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    # Fit a second order polynomial to each using `np.polyfit`
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### calculation of R_curve (radius of curvature) #####
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad, fit_cr

def measure_offset_real(img_shape, left_fit, right_fit):
    y = ym_per_pix * img_shape[0]
    l_fitValue = left_fit[0]* y**2 + left_fit[1]*y + left_fit[2]
    r_fit_Value = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    lane_center_pos = (l_fitValue + r_fit_Value) /2
    
    return lane_center_pos - img_shape[1] / 2 * xm_per_pix