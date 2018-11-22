
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time
# numpy and scipy
import numpy as np
# OpenCV
import cv2
# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

from fillter import get_processed_img
from make_decide import make_decide

stime = time.time()

# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (320,240))

class image_feature:

    def __init__(self):
        
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.speed = rospy.Publisher("/Team1_speed",Float32,queue_size=1)
        self.angle = rospy.Publisher("/Team1_steerAngle",Float32,queue_size=1)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/Team1_image/compressed",
            CompressedImage, self.callback,  queue_size = 1, buff_size=230400)
        
       

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        global stime
        #print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        binary,roi,eyeBird=get_processed_img(image_np)
        
        angle,speed=make_decide(eyeBird)

        self.speed.publish(3)
        if angle is not None and speed is not None:
            self.speed.publish(speed)
            self.angle.publish(angle)

        binary_img = np.dstack((binary, binary, binary))*255
        roi_img = np.dstack((roi, roi, roi))*255
        eyeBird_img = np.dstack((eyeBird, eyeBird, eyeBird))*255

        cv2.imshow('threshold',binary_img)
        cv2.imshow('roi',roi_img)
        cv2.imshow('eyeBird',eyeBird_img)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        stime = time.time()
        cv2.waitKey(1)

def main(args):
    '''Initializes and cleanup ros node'''
   
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
