import cv2
import numpy as np
from sliding_windown import *
from roi_lane import *
def detect_lane(img):
#print (img.shape)
    #img = cv2.imread ("3.png")
    #img = cv2.blur(img,(1,1))
    img_HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    low_threshold= (0,90,15)
    high_threshold= (58,255,255)
    frame_threshold = cv2.inRange(img_HSV,low_threshold,high_threshold)
    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(frame_threshold,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    dilation = cv2.dilate(dilation,kernel,iterations = 1)
    frame_threshold = cv2.erode(dilation,kernel,iterations = 1)
    #cv2.namedWindow('frame_threshold', cv2.WINDOW_NORMAL)
    cv2.imshow('source',img)
    cv2.imshow('lay nguong',frame_threshold)
    #img_roi = region_of_interest(frame_threshold)
    img_transform = perspective_transform(frame_threshold)
    #cv2.imshow('img_roi', img_roi)
    #img_cut = img_transform[int(imshape[0]*0.5):imshape[0],0:imshape[1]]
    ret,img_binary = cv2.threshold(img_transform,100,255,cv2.THRESH_BINARY)
    #cv2.imwrite('a.png',img_cut)
    cv2.imshow('img_transform',img_transform)
    #cv2.imshow('transform',img_transform)
    left_fit,right_fit,left_lane_inds, right_lane_inds,angle = sliding_window(img_binary)
    array_left_fitx,array_right_fitx = poly_fit(img_binary,left_fit,right_fit,left_lane_inds, right_lane_inds)
    #cv2.imshow('out_put',img_out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return angle*0.4
