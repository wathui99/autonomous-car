import numpy as np
import cv2
from nhan_dien_duong_line import detect_line
from geometry import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


def make_decide (binary_img,threshold_num_point=4):

	midPoint_img = np.dstack((binary_img, binary_img, binary_img))*255 #cho hien thi anh

	left_line,right_line=detect_line(binary_img,)

	if (left_line is None) and (right_line is None):
		print ('bad line')
		return None,None

	suitLines = suit_lines(left_line,right_line)
	mid_points = list_mid_points(suitLines)
	mid_point = average_mid_point(mid_points)

	if(mid_point is None):
		print('bad midPoint')
		return None,None

	cv2.circle(midPoint_img,(int(mid_point[0]),int(mid_point[1])), 5, (0,0,255), -1)

	cv2.imshow('mid_point', midPoint_img)

	ImgShapeX=binary_img.shape[1]
	ImgShapeY=binary_img.shape[0]

	angle,speed = caculate_angle_speed (ImgShapeX=ImgShapeX,ImgShapeY=ImgShapeY,midPoint=mid_point,ratioAngle=1,maxSpeed=70,ratioSpeed=1)
	#print(angle,speed)
	return angle,speed