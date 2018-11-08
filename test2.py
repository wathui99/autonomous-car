#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from sliding import *

analyzed=None

def for_sliding_window(binary_warped):
    global analyzed
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    #histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    slidingPice=16
    for iHistogram in range(1,slidingPice+1):
        #first=binary_warped.shape[0]*(iHistogram-1)/slidingPice
        #second=binary_warped.shape[0]*(iHistogram)/slidingPice
        #print ("==============first, second==============")
        #print (first,second)
    	histogram = np.sum(binary_warped[binary_warped.shape[0]*(iHistogram-1)/slidingPice:binary_warped.shape[0]*(iHistogram)/slidingPice,:], axis=0)
    	#print ("==============histogram==============")
        #print (histogram)
        if iHistogram==1:
            analyzed=np.array([analyze_histogram (histogram,windowLength=4,emptyLength=10,thresholdEmpty=0,thresholdSum=5)])
        else:
            temp = np.array([analyze_histogram (histogram,windowLength=4,emptyLength=10,thresholdEmpty=0,thresholdSum=5)])
            analyzed = np.append(analyzed,temp,axis=0)
    analyzed,countDetectedLeft,countDetectedRight = rearrange_analyze_histogram (analyzed,slidingPice,thresholdGapLine=7)
    print (analyzed)
    return analyzed

def nothing(x):
    pass

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def color_thr(s_img, v_img, s_threshold = (0,255), v_threshold = (0,255)):
    s_binary = np.zeros_like(s_img).astype(np.uint8)
    s_binary[(s_img > s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    v_binary = np.zeros_like(s_img).astype(np.uint8)
    v_binary[(v_img > v_threshold[0]) & (v_img <= v_threshold[1])] = 1
    col = ((s_binary == 1) | (v_binary == 1))
    return col

#the main thresholding operaion is performed here 
def thresholding(img, s_threshold_min = 113, s_threshold_max = 255, v_threshold_min = 234, v_threshold_max = 255,  k_size = 5):
    # Convert to HSV color space and separate the V channel
    imshape = img.shape
    #convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #read saturation channel
    s_channel = hls[:,:,2].astype(np.uint8)
    #Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    #read the value channel
    v_channel = hsv[:,:,2].astype(np.uint8)
    #threshold value and saturation
    threshold_binary = color_thr(s_channel, v_channel, s_threshold=(s_threshold_min,s_threshold_max), v_threshold= (v_threshold_min,v_threshold_max)).astype(np.uint8)
    #caculate ROI
    lower_left=[0,imshape[0]]
    lower_right=[imshape[1],imshape[0]]
    top_left=[.15*imshape[1], 0.4*imshape[0]]
    top_right=[imshape[1]-.15*imshape[1], 0.4*imshape[0]]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_binary=region_of_interest(threshold_binary,vertices)
    #for display threshold sv
    threshold_img = np.dstack((threshold_binary, threshold_binary, threshold_binary))*255
    cv2.imshow('threshold_sv',threshold_img)
    #for display roi
    roi_img = np.dstack((roi_binary, roi_binary, roi_binary))*255
    cv2.imshow('roi',roi_img)
    #for display perspective transform
    #perspective transform
    per_binary = perspective_transform(roi_binary)

    per_img = np.dstack((per_binary, per_binary, per_binary))*255
    cv2.imshow('per',per_img)
    return per_binary

#perspective transform on undistorted images
def perspective_transform(img):
    imshape = img.shape
    #print (img.shape[0])
    lower_left=[0,imshape[0]]
    lower_right=[imshape[1],imshape[0]]
    top_left=[.41*imshape[1], 0.4*imshape[0]]
    top_right=[imshape[1]-.41*imshape[1], 0.4*imshape[0]]

    vertices = np.array([[tuple(top_right), tuple(lower_right),
                       tuple(lower_left),tuple(top_left)]], dtype=np.float32)
    src= np.float32(vertices)
    dst = np.float32([[0.6*img.shape[1],0],[0.6*img.shape[1],img.shape[0]],
                      [0.4*img.shape[1],img.shape[0]],[0.4*img.shape[1],0]])
    #print (dst)
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (imshape[1], imshape[0]) 
    perspective_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)    
    return perspective_img


cap = cv2.VideoCapture('/home/lee/Downloads/outpy.avi')
cv2.namedWindow('trackbar')
cv2.createTrackbar('min_v','trackbar',0,255,nothing)
cv2.createTrackbar('max_v','trackbar',0,255,nothing)
cv2.createTrackbar('min_s','trackbar',0,255,nothing)
cv2.createTrackbar('max_s','trackbar',0,255,nothing)
while(cap.isOpened()):
    ret, img = cap.read()
    min_v = cv2.getTrackbarPos('min_v','trackbar')
    max_v = cv2.getTrackbarPos('max_v','trackbar')
    min_s = cv2.getTrackbarPos('min_s','trackbar')
    max_s = cv2.getTrackbarPos('max_s','trackbar')
    if img is not None:
    	binary_result=thresholding(img,230,240,220,255) #230,240,220,255
    	for_sliding_window(binary_result)
    	cv2.imshow('raw',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
