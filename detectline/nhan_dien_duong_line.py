import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fillter import get_processed_img
import time
import math

#co gang lap day cac lo trong:
#dua vao diem dau va diem cuoi
def fill_points(points,distance_2Points=2):
	first_point_flag=False
	first_point=-1
	pos_first_point=-1
	conti=False
	for i in range(points.shape[0]):

		if (points[i][0] != -1) and (first_point_flag==False):
			first_point_flag = True
			pos_first_point = i
			first_point = points[i][0]
		if (points[i][0] != -1) and first_point_flag:
			if pos_first_point == i:
				continue
			for iPoint in range(distance_2Points):
				if (i+iPoint) >= points.shape[0]:
					conti = True
					break
				if (points[i+iPoint][0] == -1):
					first_point_flag=True
					pos_first_point = i
					first_point = points[i][0]
					continue

			if conti:
				conti=False
				continue

			if (i-pos_first_point) == 0:
				first_point_flag=False
				continue

			distance = (points[i][0] - first_point) / (i-pos_first_point)
			
			for j in range(pos_first_point+1, i):
				points[j][0] = points[j-1][0] + distance
			first_point_flag=False
			
#tim diem dau tien de bat dau tim line
#arr la histogram
#startPos la diem bat dau tim
#endPos la gioi han tim kiem
#left_or_right:
  #0 -> tim line trai
  #1 -> tim line phai
#step cang cao toc do cang nhanh -> chinh xac giam

def find_first_point (arr, startPos, endPos, left_or_right, windowLength_line=7, windowLength_empty=20, step=2, threshold_empty=15, threshold_line=50):
	#ben phai vi tri lan luot la:
	# ....00000012321....
	if left_or_right == 1: #nhan dien ben phai		
		for iPos in range (0,endPos-startPos,step):
			#diem bat dau
			#tim phan khong co line truoc
			if (startPos+iPos+windowLength_empty-1) >= endPos:
				#vuot qua gioi han -> thoat luon
				return None,None
			sum_window_empty = np.sum(arr[startPos+iPos:startPos+iPos+windowLength_empty])
			#tim phan co line
			sum_wimdow_line=0
			if (startPos+iPos+windowLength_empty+windowLength_line-1 >= endPos):
				sum_wimdow_line=np.sum(arr[startPos+iPos+windowLength_empty:endPos])
				if sum_wimdow_line < threshold_line:
					#cham bien nhung khong tim thay line => thoat luon
					return None,None
			else:
				sum_wimdow_line=np.sum(arr[startPos+iPos+windowLength_empty:startPos+iPos+windowLength_empty+windowLength_line])
			
			if (sum_window_empty <= threshold_empty) and (sum_wimdow_line >= threshold_line):
				return (startPos+iPos+windowLength_empty-1) + int(windowLength_line/2), sum_wimdow_line
	
	#ben trai vi tri lan luot la:
	# ....12321000000....
	if left_or_right == 0: #nhan dien ben trai
		for iPos in range (0,endPos-startPos,step):
			#diem bat dau
			#tim phan khong co line truoc
			if (startPos+iPos+windowLength_line+windowLength_empty-1) >= endPos:
				#vuot qua gioi han -> thoat luon
				return None,None
			sum_window_empty = np.sum(arr[startPos+iPos+windowLength_line : startPos+iPos+windowLength_empty+windowLength_line])
			#tim phan co line
			sum_wimdow_line=0
			if (startPos+iPos+windowLength_line-1 >= endPos):
				sum_wimdow_line=np.sum(arr[startPos+iPos:endPos])
				if sum_wimdow_line < threshold_line:
					#cham bien nhung khong tim thay line => thoat luon
					return None,None
			else:
				sum_wimdow_line=np.sum(arr[startPos+iPos:startPos+iPos+windowLength_line])
			
			if (sum_window_empty <= threshold_empty) and (sum_wimdow_line >= threshold_line):
				return (startPos+iPos) + int(windowLength_line/2), sum_wimdow_line
	return None,None

def find_point_in_line (arr,start_point, left_or_right, step=1, distance_2Points=10, threshold_line=13, windowLength_line=7, threshold_empty=5, windowLength_empty=20):
	width = arr.shape[0]
	#line trai ...1230000...
	if left_or_right == 0: #line trai
		for iPos in range(0,distance_2Points+1,step):
			#side ben phai line==================================================

			#tinh empty
			#tranh vuot qua bien ben phai
			if (start_point+iPos+int(windowLength_line/2)+windowLength_empty) < width:
				#tong diem bat dau tinh empty => diem bat dau + windowLength_empty 
				sum_window_empty=np.sum(arr[start_point+iPos+int(windowLength_line/2)+1 : start_point+iPos+int(windowLength_line/2)+1+windowLength_empty])

				#tinh diem line
				sum_wimdow_line=0
				#vuot qua bien ben trai
				if (start_point+iPos-int(windowLength_line/2)) < 0:
					sum_wimdow_line=np.sum(arr[0 : start_point+iPos+int(windowLength_line/2)+1])
				else:
					sum_wimdow_line=np.sum(arr[start_point+iPos-int(windowLength_line/2) : start_point+iPos+int(windowLength_line/2)+1])
				if (sum_window_empty <= threshold_empty) and (sum_wimdow_line >= threshold_line):
					return start_point+iPos

			#side ben trai line===================================================

			#tinh empty
			#tranh vuot qua bien ben phai
			if (start_point-iPos+int(windowLength_line/2)+windowLength_empty) < width:
				#tong diem bat dau tinh empty => diem bat dau + windowLength_empty 
				sum_window_empty=np.sum(arr[start_point-iPos+int(windowLength_line/2)+1 : start_point-iPos+int(windowLength_line/2)+1+windowLength_empty])

				#tinh diem line
				#vuot qua bien ben trai
				sum_wimdow_line=0
				if (start_point-iPos-int(windowLength_line/2)) < 0:
					sum_wimdow_line=np.sum(arr[0 : start_point-iPos+int(windowLength_line/2)+1])
				else:
					sum_wimdow_line=np.sum(arr[start_point-iPos-int(windowLength_line/2) : start_point-iPos+int(windowLength_line/2)+1])
				if (sum_window_empty <= threshold_empty) and (sum_wimdow_line >= threshold_line):
					return start_point-iPos
	#line phai ...0000123...
	if left_or_right == 1: #line phai
		for iPos in range(0,distance_2Points+1,step):
			#side ben phai line==================================================

			#tinh empty
			start_pos_empty = start_point+iPos-int(windowLength_line/2)-windowLength_empty
			end_pos_empty = start_pos_empty+windowLength_empty
			#tranh vuot qua bien ben trai
			if (start_pos_empty < 0):
				start_pos_empty=0
			
			sum_window_empty=0
			#tranh vuot bien phai
			if (end_pos_empty <= width) and (end_pos_empty>=0):
				sum_window_empty = np.sum(arr[start_pos_empty:end_pos_empty])
				#neu thoa dieu kien khong phai line
				if (sum_window_empty <= threshold_empty):
					start_pos_line=end_pos_empty
					end_pos_line = start_pos_line + windowLength_line
					if end_pos_line > width:
						end_pos_line = width
					if (end_pos_line > start_pos_line):
						sum_wimdow_line = np.sum(arr[start_pos_line:end_pos_line])
						if (sum_wimdow_line >= threshold_line):
							return int((end_pos_line - start_pos_line)/2+start_pos_line)

			#side ben trai line==================================================

			#tinh empty
			start_pos_empty = start_point-iPos-int(windowLength_line/2)-windowLength_empty
			end_pos_empty = start_pos_empty+windowLength_empty
			#tranh vuot qua bien ben trai
			if (start_pos_empty < 0):
				start_pos_empty=0
			
			sum_window_empty=0
			#tranh vuot bien phai
			if (end_pos_empty <= width) and (end_pos_empty>=0):
				sum_window_empty = np.sum(arr[start_pos_empty:end_pos_empty])
				#neu thoa dieu kien khong phai line
				if (sum_window_empty <= threshold_empty):
					start_pos_line=end_pos_empty
					end_pos_line = start_pos_line + windowLength_line
					if end_pos_line > width:
						end_pos_line = width
					if (end_pos_line > start_pos_line):
						sum_wimdow_line = np.sum(arr[start_pos_line:end_pos_line])
						if (sum_wimdow_line >= threshold_line):
							return int((end_pos_line - start_pos_line)/2+start_pos_line)
					
	#deo' tim ra diem nao thoa
	return -1

#tim cac vi tri co the xuat hien cua duong line

def find_line (eyeBird_binary,nWindows,first_point, left_or_right, bottom_or_top):
	img_heigh = eyeBird_binary.shape[0]
	img_width = eyeBird_binary.shape[1]
	#chieu cao cua window
	window_heigh = int(img_heigh/nWindows)

	pre_line = first_point
	pre_line_pos = 1 #dung de gia tang pham vi tim kiem

	points_line=None
	nPoints_line=0

	for iWindow in range(1,nWindows+1):
		#neu tim tu bottom len
		sub_his=None
		if bottom_or_top == 0:
			sub_his = np.sum(eyeBird_binary[img_heigh - iWindow*window_heigh:img_heigh - (iWindow - 1)*window_heigh,:], axis=0)
		#neu tim tu top xuong
		if bottom_or_top == 1:
			sub_his = np.sum(eyeBird_binary[(iWindow-1)*window_heigh:iWindow*window_heigh,:], axis=0)
		x=find_point_in_line (arr=sub_his,start_point=pre_line, left_or_right=left_or_right, 
								step=2, distance_2Points=30+(iWindow-pre_line_pos)*2, 
								threshold_line=8, windowLength_line=7, 
								threshold_empty=3, windowLength_empty=20)
		if points_line is None:
			if bottom_or_top == 0:
				y = img_heigh - window_heigh*iWindow
			if bottom_or_top == 1:
				y = window_heigh*iWindow
			points_line=np.array([[x,y]])
		else:
			if bottom_or_top == 0:
				y = img_heigh - window_heigh*iWindow
			if bottom_or_top == 1:
				y = window_heigh*iWindow
			coo = np.array([[x,y]])
			points_line = np.append(points_line,coo,axis=0) #them vao left line
		#cap nhat vi tri line moi
		if (x!=-1): #neu tim thay vi tri moi thi -> cap nhat
			pre_line = x
			pre_line_pos = iWindow
			nPoints_line += 1
	if (bottom_or_top==0):
		return points_line,nPoints_line
	if (bottom_or_top==1):
		#temp = points_line[::-1]
		return points_line[::-1],nPoints_line

#case = 0: ben trai truoc
#case = 1: ben phai truoc

#return left_line, right_line
#(None,None),(None,right_line),(left_line,None): neu khong nhan dien duoc diem dau

def get_line (eyeBird_binary,case,threshold_num_point=5):
	img_heigh = eyeBird_binary.shape[0]
	img_width = eyeBird_binary.shape[1]
	if case == 0:
		#lay tong cua nua phan duoi
		half_bottom_histogram = np.sum(eyeBird_binary[int(img_heigh/2):img_heigh,:], axis=0)
		#tim vi tri dau tien line ben trai
		#nua duoi ben trai -> 1/2
		left_x,_=find_first_point (arr=half_bottom_histogram, 
				startPos=50, 
				endPos=int(img_width/2)+50, left_or_right=0, 
				windowLength_line=7, windowLength_empty=20, step=2, 
				threshold_empty=15, threshold_line=50)
		if left_x is not None:
			#nua duoi ke tu line trai qua phai -> cuoi
			right_x,_=find_first_point (arr=half_bottom_histogram, 
					startPos=left_x, 
					endPos=img_width, left_or_right=1, 
					windowLength_line=7, windowLength_empty=20, step=2, 
					threshold_empty=15, threshold_line=50)
			#tim thay diem bat dau line phai
			if right_x is not None:
				#do line phai tu duoi len
				right_line, nRight = find_line (eyeBird_binary=eyeBird_binary,nWindows=16,first_point=right_x, 
							left_or_right=1, bottom_or_top=0)
				#do line trai tu duoi len
				left_line, nLeft = find_line (eyeBird_binary=eyeBird_binary,nWindows=16,first_point=left_x, 
							left_or_right=0, bottom_or_top=0)
				#kiem tra xem so diem do duoc co thoa threshold_num_point hay ko
				if nRight >= threshold_num_point and nLeft >= threshold_num_point:
					fill_points(left_line)
					fill_points(right_line)
					return left_line,right_line
				if nRight >= threshold_num_point:
					#fill_points(right_line)
					return None,right_line
				if nLeft >= threshold_num_point:
					#fill_points(left_line)
					return left_line,None

	if case == 1:
		#lay tong cua nua phan duoi
		half_bottom_histogram = np.sum(eyeBird_binary[int(img_heigh/2):img_heigh,:], axis=0)
		#tim vi tri dau tien line ben trai
		#nua duoi ben trai -> 1/2
		right_x,_=find_first_point (arr=half_bottom_histogram, 
				startPos=int(img_width/2)-20, 
				endPos=img_width, left_or_right=1, 
				windowLength_line=7, windowLength_empty=20, step=2, 
				threshold_empty=15, threshold_line=50)
		if right_x is not None:
			#nua duoi ke tu line trai qua phai -> cuoi
			left_x,_=find_first_point (arr=half_bottom_histogram, 
					startPos=0, 
					endPos=right_x, left_or_right=0, 
					windowLength_line=7, windowLength_empty=20, step=2, 
					threshold_empty=15, threshold_line=50)
			#tim thay diem bat dau line phai
			if left_x is not None:
				#do line phai tu duoi len
				right_line, nRight = find_line (eyeBird_binary=eyeBird_binary,nWindows=16,first_point=right_x, 
							left_or_right=1, bottom_or_top=0)
				#do line trai tu duoi len
				left_line, nLeft = find_line (eyeBird_binary=eyeBird_binary,nWindows=16,first_point=left_x, 
							left_or_right=0, bottom_or_top=0)
				#kiem tra xem so diem do duoc co thoa threshold_num_point hay ko
				if nRight >= threshold_num_point and nLeft >= threshold_num_point:
					fill_points(left_line)
					fill_points(right_line)
					return left_line,right_line
				if nRight >= threshold_num_point:
					#fill_points(right_line)
					return None,right_line
				if nLeft >= threshold_num_point:
					#fill_points(left_line)
					return left_line,None

	#khong thoa bat cu dieu gi
	return None,None

#tra ve 1 mang gom left line va right line
#
# left = ([[xleft,yleft],[x2,y2],...])
# right = ([[xright,yright],[x2,y2],...]), .... nWindows

def detect_line(eyeBird_binary,threshold_num_point=8):

	line_detect = np.dstack((eyeBird_binary, eyeBird_binary, eyeBird_binary))*255

	left_line, right_line = get_line (eyeBird_binary,case=0,threshold_num_point=threshold_num_point)

	if (left_line is not None) and (right_line is not None):
		for point_right, point_left in zip(right_line,left_line):
			cv2.circle(line_detect,(point_right[0],point_right[1]), 5, (0,255,0), -1)
			cv2.circle(line_detect,(point_left[0],point_left[1]), 5, (0,0,255), -1)

	else:
		left_line, right_line = get_line (eyeBird_binary,case=1,threshold_num_point=threshold_num_point)
		if (left_line is not None) and (right_line is not None):
			for point_right, point_left in zip(right_line,left_line):
				cv2.circle(line_detect,(point_right[0],point_right[1]), 5, (0,255,0), -1)
				cv2.circle(line_detect,(point_left[0],point_left[1]), 5, (0,0,255), -1)
	
	cv2.imshow('line_detect',line_detect)
	
	if left_line is None or right_line is None:
		return None,None
	fill_line_left=None
	fill_line_right=None
	for i in range(left_line.shape[0]):
		if (left_line[i][0] != -1):
			if (fill_line_left is not None):
				temp=np.array([[left_line[i][0],left_line[i][1]]])
				fill_line_left=np.append(fill_line_left,temp,axis=0)
			else:
				fill_line_left=np.array([[left_line[i][0],left_line[i][1]]])
		if (right_line[i][0] != -1):
			if (fill_line_right is not None):
				temp=np.array([[right_line[i][0],right_line[i][1]]])
				fill_line_right=np.append(fill_line_right,temp,axis=0)
			else:
				fill_line_right=np.array([[right_line[i][0],right_line[i][1]]])
	return fill_line_left,fill_line_right

cap = cv2.VideoCapture('outpy.avi')
if __name__ == '__main__':
	n=0
	#arr=np.array([[-1,0],[2,1],[3,2],[-1,3],[-1,4],[6,5],[-1,6],[-1,7]])
	#fill_points(arr)
	#print (arr)
	side=-1
	while(cap.isOpened()):
		stime = time.time()
		ret, image_np = cap.read()
		#image_np=cv2.imread('difficult.png')
		res_binary,roi_binary,eyeBird_binary=get_processed_img(image_np)
		img_heigh = eyeBird_binary.shape[0]
		img_width = eyeBird_binary.shape[1]
		
		detect_line(eyeBird_binary=eyeBird_binary,threshold_num_point=8)

		cv2.imshow('raw',image_np)
		#print ('{} FPS'.format(1.0/(time.time() - stime)))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
