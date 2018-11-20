import numpy as np
import cv2
from get_processed_img import process_img
from nhan_dien_duong_line import detect_line
from geometry import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


def make_decide (img,convert2RGB=True):
	img_raw = None

	if convert2RGB:
		img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	else:
		img_raw = img

	thres_binary, roi_binary, eyeBird_binary = process_img(img_raw)

	thres_img = np.dstack((thres_binary, thres_binary, thres_binary))*255 #cho hien thi anh
	roi_img = np.dstack((roi_binary, roi_binary, roi_binary))*255 #cho hien thi anh
	eyeBird_img = np.dstack((eyeBird_binary, eyeBird_binary, eyeBird_binary))*255 #cho hien thi anh
	midPoint_img = np.dstack((eyeBird_binary, eyeBird_binary, eyeBird_binary))*255 #cho hien thi anh


	cv2.imshow('thres',thres_img)
	cv2.imshow('roi',roi_img)
	cv2.imshow('eyeBird', eyeBird_img)

	left_line,right_line=detect_line(eyeBird_binary)

	if left_line is None:
		print('bad left line')
	if right_line is None:
		print('bad right line')

	suitLines = suit_lines(left_line,right_line)
	mid_points = list_mid_points(suitLines)
	mid_point = average_mid_point(mid_points)

	if(mid_point is None):
		print('bad result')
		return None

	cv2.circle(midPoint_img,(int(mid_point[0]),int(mid_point[1])), 5, (0,0,255), -1)

	cv2.imshow('mid_point', midPoint_img)

	ImgShapeX=eyeBird_binary.shape[1]
	ImgShapeY=eyeBird_binary.shape[0]

	angle=speed=None
	if mid_point is None:
		print('bad result')
		return angle,speed #return None
	else:
		angle,speed = caculate_angle_speed (ImgShapeX=ImgShapeX,ImgShapeY=ImgShapeY,midPoint=mid_point,ratioAngle=1,maxSpeed=60,ratioSpeed=1)
	print(angle,speed)
	return angle,speed

cap = cv2.VideoCapture('outpy.avi')

if __name__ == '__main__':
	while(cap.isOpened()):
		stime = time.time()
		#bgr_img = cv2.imread('3840.png')
		ret, frame = cap.read()
		if ret == False:
			continue
		cv2.imshow('raw',frame)
		make_decide (frame,convert2RGB=True)
		print('FPS {:.1f}'.format(1 / (time.time() - stime)))
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()
