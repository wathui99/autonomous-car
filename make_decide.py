import numpy as np
import cv2
from nhan_dien_duong_line import detect_line, detect_one_line
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

	angle,speed = caculate_angle_speed (ImgShapeX=ImgShapeX,ImgShapeY=ImgShapeY,midPoint=mid_point,ratioAngle=1,maxSpeed=65,ratioSpeed=1)

	#print(angle,speed)
	return angle,speed

def follow_one_line (binary_img,left_or_right):
	ImgShapeX=binary_img.shape[1]
	ImgShapeY=binary_img.shape[0]

	midPoint_img = np.dstack((binary_img, binary_img, binary_img))*255 #cho hien thi anh
	if (left_or_right==0):
		left_line,nLeft=detect_one_line(binary_img,left_or_right=0) #left
		#print (left_line,nLeft)
		if (nLeft > 0):
			sumX=left_line[0][0] #gan khoi tao cho vi tri dau tien
			posY=left_line[0][1] #gan cho vi tri cao nhat
			#lay posY nam trong vung tu 1/3<=>2/3 chieu cao anh
			if (posY < int(1/3*ImgShapeY)):
				posY=int(1/2*ImgShapeY) #gan thang posY o vi tri trung tam
			else:
				posY=int(2/3*ImgShapeY)
			for i in range(1,nLeft):
				sumX+=left_line[i][0]
			posX=int(sumX/nLeft)
			posX+=47 #vi tri tu line den trung diem
			print (posY,posX)
			mid_point=np.array([posX,posY])
			cv2.circle(midPoint_img,(int(mid_point[0]),int(mid_point[1])), 5, (0,0,255), -1)
			angle,speed = caculate_angle_speed (ImgShapeX=ImgShapeX,ImgShapeY=ImgShapeY,midPoint=mid_point,ratioAngle=1,maxSpeed=40,ratioSpeed=1)
			cv2.imshow('mid_point', midPoint_img)
			return angle,speed

	if (left_or_right==1):
		right_line,nRight=detect_one_line(binary_img,left_or_right=1) #right
		print (right_line,nRight)
		if (nRight>0):
			sumX=right_line[0][0] #gan khoi tao cho vi tri dau tien
			posY=right_line[0][1] #gan cho vi tri cao nhat
			#lay posY nam trong vung tu 1/3<=>2/3 chieu cao anh
			if (posY < int(1/3*ImgShapeY)):
				posY=int(1/2*ImgShapeY) #gan thang posY o vi tri trung tam
			else:
				posY=int(2/3*ImgShapeY)
			for i in range(1,nRight):
				sumX+=right_line[i][0]
			posX=int(sumX/nRight)
			posX-=47 #vi tri tu line den trung diem
			print (posY,posX)
			mid_point=np.array([posX,posY])
			cv2.circle(midPoint_img,(int(mid_point[0]),int(mid_point[1])), 5, (0,0,255), -1)
			angle,speed = caculate_angle_speed (ImgShapeX=ImgShapeX,ImgShapeY=ImgShapeY,midPoint=mid_point,ratioAngle=1,maxSpeed=40,ratioSpeed=1)
			cv2.imshow('mid_point', midPoint_img)
			return angle,speed
	cv2.imshow('mid_point', midPoint_img)
	return None,None