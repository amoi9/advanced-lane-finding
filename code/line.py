import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, last_n=10):
        # Detected frame
        self.frame = 0
        # was the line detected in the last iteration?
        self.detected = False  
        # last N
        self.last_n = last_n
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
    
    def set_params(self, current_fit, radius_of_curvature, line_base_pos, allx, frame):
        self.detected = True
        self.current_fit = current_fit
        self.radius_of_curvature = radius_of_curvature
        self.line_base_pos = line_base_pos
        self.allx = allx
        self.frame = frame
        if len(self.recent_xfitted) == 0:
            self.recent_xfitted = np.empty((0,len(allx)), float)
        self.recent_xfitted = np.append(self.recent_xfitted, np.array([allx]), axis=0)
        self.bestx = np.average(self.recent_xfitted, axis=0)
        if len(self.recent_xfitted) > self.last_n:
            np.delete(self.recent_xfitted, 0)
        