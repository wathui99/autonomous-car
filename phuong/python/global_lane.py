import numpy as np
global left_lane 
global right_lane
global Point_Center
global Point_xe
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self): 
        # was the line detected in the last iteration?
        self.detected = False  
         # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = np.zeros(480)
         #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.zeros(3)  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = np.zeros(1)
        #distance in meters of vehicle center from the line
        self.line_base_pos = np.zeros(1)
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #smoothen the n frames 
        self.smoothen_nframes = 10
        #first frame 
        self.first_frame = True
class Point:
    """ Point class represents and manipulates x,y coords. """
    def __init__(self, x=0, y=0):
        """ Create a new point at x, y """
        self.x = x
        self.y = y
left_lane = Line()
right_lane = Line()
Point_Center= Point()
Point_xe= Point()

