
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

from darkflow.net.build import TFNet

from fillter import get_processed_img
from make_decide import make_decide, follow_one_line

stime = time.time()

options = {
	'model': 'cfg/tiny-yolo-voc-3c.cfg',
	'load': 30000,
	'threshold': 0.2,	
	'gpu': 0.5
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (320,240))

turn_left=0
turn_right=0
none=0
step=0 #0 la do 2 line, 1 la do 1 line (tin hieu re)
time_start_turn=0

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
        global turn_left
        global turn_right
        global none
        global stime
        global time_start_turn
        global step
        #print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

        results = tfnet.return_predict(image_np)

        binary,roi,eyeBird=get_processed_img(img=image_np)

        if (step==1 and time.time()-time_start_turn>=1.0): #turning complete
            step=0 #normal mode

        if (turn_left>=10 or turn_right>=10):
            if (turn_right>=10): #tin hieu re phai
                angle,speed=follow_one_line (binary_img=eyeBird,left_or_right=1)
                if angle is not None and speed is not None:
                    self.speed.publish(speed)
                    self.angle.publish(angle)
                    none=0
                else:
                    none+=1
                    if (none>=5): #re phai gap
                        self.speed.publish(50)
                        self.angle.publish(30)
                        turn_right=0
                        turn_left=0
                        none=0
                        time_start_turn=time.time() #time count
                        step=1 #turning
                        print ('turning right')
            if (turn_left>=10): #tin hieu re phai
                angle,speed=follow_one_line (binary_img=eyeBird,left_or_right=0)
                if angle is not None and speed is not None:
                    self.speed.publish(speed)
                    self.angle.publish(angle)
                    none=0
                else:
                    none+=1
                    if (none>=5): #re trai gap
                        self.speed.publish(50)
                        self.angle.publish(-30)
                        turn_right=0
                        turn_left=0
                        none=0
                        time_start_turn=time.time() #time count
                        step=1 #turning
                        print ('turning left')
        else:
            if (step==0): #normal mode
                angle,speed=make_decide(eyeBird)
                self.speed.publish(40)
                if angle is not None and speed is not None:
                    self.speed.publish(speed)
                    self.angle.publish(angle)

        
        #angle,speed=make_decide(eyeBird)

        #self.speed.publish(3)
        #if angle is not None and speed is not None:
            #self.speed.publish(speed)
            #self.angle.publish(angle)
            #none=0
        #else:
        	#none+=1

        binary_img = np.dstack((binary, binary, binary))*255
        roi_img = np.dstack((roi, roi, roi))*255
        eyeBird_img = np.dstack((eyeBird, eyeBird, eyeBird))*255

        for color, result in zip(colors,results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            if result['label'] == 'turn_right':
            	turn_right+=1
            if result['label'] == 'turn_left':
                turn_left+=1
            confidence = result['confidence']
            text = '{}:{:.0f}%'.format(label,confidence*100)
            frame = cv2.rectangle(image_np,tl,br,color,1)
            frame = cv2.putText(image_np,text,tl,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        
        #if (angle is None) and (speed is None):
            #if turn_left >= 5 and none >= 1:
                #self.speed.publish(50)
            	#self.angle.publish(-50)
            	#turn_left = 0
            	#turn_right = 0
            #if turn_right >= 5 and none >=1:
                #self.speed.publish(50)
            	#self.angle.publish(50)
            	#turn_left = 0
            	#turn_right = 0
        cv2.imshow('raw',image_np)
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
