import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fillter import get_processed_img
import time


#tinh tong max va tim vi tri max o giua cua mang con
def max_sub_array (arr, startPos, endPos, windowLength):
	max_sub = np.sum(arr[startPos:startPos+windowLength])
	current_sum = max_sub
	max_sub_pos = startPos
	for i in range (startPos+1,endPos-windowLength+2):
		current_sum = current_sum + arr[i+windowLength-1] - arr[i-1] #cong dau tru cuoi
		if (current_sum > max_sub):
			max_sub=current_sum
			max_sub_pos = i + int(windowLength/2)
	return max_sub,max_sub_pos

#co gang lap day cac lo trong:
#dua vao diem dau va diem cuoi
def fill_points(points):
	empty_point = False
	pos_before = -1 #vi tri ko phai lo trong truoc vi tri lo trong
	point_before = 0 #gia tri cua points[pos_before]
	for i in range(points.shape[0]):
		if points[i][0] == -1: #points[i][0] -> chi lay phan x
			empty_point = True
			continue
		if empty_point:
			#if points[i] != -1    {do co continue va dieu kien == -1 o tren nen khong can dong nay}
			distance = int((points[i][0] - point_before) / (i-pos_before))
			for j in range(pos_before+1, i):
				points[j][0] = points[j-1][0] + distance
			empty_point = False
		else:
			point_before = points[i][0]
			pos_before = i

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



def find_line (binary,nWindows,first_point, left_or_right, bottom_or_top):
	img_heigh = eyeBird_binary.shape[0]
	img_width = eyeBird_binary.shape[1]
	#chieu cao cua window
	window_heigh = int(img_heigh/nWindows)

	pre_line = first_point
	pre_line_pos = 1 #dung de gia tang pham vi tim kiem

	points_line=None

	for iWindow in range(1,nWindows+1):
		#neu tim tu bottom len
		sub_his=None
		if bottom_or_top == 0:
			sub_his = np.sum(eyeBird_binary[img_heigh - iWindow*window_heigh:img_heigh - (iWindow - 1)*window_heigh,:], axis=0)
		#neu tim tu top xuong
		if bottom_or_top == 1:
			sub_his = np.sum(eyeBird_binary[(iWindow-1)*window_heigh:iWindow*window_heigh,:], axis=0)
		x=find_point_in_line (arr=sub_his,start_point=pre_line, left_or_right=left_or_right, 
								step=1, distance_2Points=10+(iWindow-pre_line_pos)*2, 
								threshold_line=9, windowLength_line=7, 
								threshold_empty=5, windowLength_empty=20)
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

	return points_line

		

#tra ve 1 mang gom left line va right line
#
# left = ([[xleft,yleft],[x2,y2],...])
# right = ([[xright,yright],[x2,y2],...]), .... nWindows
#
#Chu y: khong nhan dien duoc tra ve None
#step: toc do truot tren histogram.  cang cao toc do duyet cang nhanh - > do chinh xac giam
def detect_line(eyeBird_binary):
	img_heigh = eyeBird_binary.shape[0]
	img_width = eyeBird_binary.shape[1]
	#lay tong cua nua phan duoi
	half_bottom_histogram = np.sum(eyeBird_binary[int(img_heigh/2):img_heigh,:], axis=0)
	#tim vi tri dau tien line ben trai
	left_first_point = find_first_point (arr=half_bottom_histogram, startPos=0, endPos=int(img_width/2), 
										left_or_right=0, windowLength_line=7, windowLength_empty=20, 
										step=2, threshold_empty=15, threshold_line=50)
	#tim vi tri dau tien line ben phai
	right_first_point = find_first_point (arr=half_bottom_histogram, startPos=0, endPos=int(img_width/2), 
										left_or_right=1, windowLength_line=7, windowLength_empty=20, 
										step=2, threshold_empty=15, threshold_line=50)

	#chieu cao cua cua so tinh tong
	window_heigh = int(img_heigh/nWindows)
	#khoi tao gia tri tra ve
	left_line = None
	right_line = None
	#lay vi tri tim line truoc do
	if (first_left_point):
		pre_line_left = max_sub_pos_left #vi tri nay theo x
		left_line = np.array([[max_sub_pos_left,img_heigh]]) #x,y ban dau cua line trai

	if (first_right_point):
		pre_line_right = max_sub_pos_right
		right_line = np.array([[max_sub_pos_right, img_heigh]]) #x,y ban dau cua line phai

	#bat dau tinh histogram tung mang va detect line
	#do line tu phia duoi di len
	for iWindow in range(1,nWindows+1):
		sub_his = np.sum(eyeBird_binary[img_heigh - iWindow*window_heigh:img_heigh - (iWindow - 1)*window_heigh,:], axis=0)
		#tinh toa do line trai truoc
		if (first_left_point):
			x=find_pos_line(histogram=sub_his, windowLength=11, threshold_line=15, 
				threshold_empty=3, length_empty=50, count_empty=20, 
				pre_linePos=pre_line_left, max_2Lines=20, sideLine=0)
			coo = np.array([[x,img_heigh - window_heigh*iWindow]])
			left_line = np.append(left_line,coo,axis=0) #them vao left line
			#cap nhat vi tri line moi
			if (x!=-1): #neu tim thay vi tri moi thi -> cap nhat
				pre_line_left = x
		#tinh toan toa do line phai
		if (first_right_point):
			x=find_pos_line(histogram=sub_his, windowLength=11, threshold_line=15, 
				threshold_empty=3, length_empty=50, count_empty=20, 
				pre_linePos=pre_line_right, max_2Lines=20, sideLine=1)
			coo = np.array([[x,img_heigh - window_heigh*iWindow]])
			right_line = np.append(right_line,coo,axis=0) #them vao left line
			#cap nhat vi tri line moi
			if (x!=-1): #neu tim thay vi tri moi thi -> cap nhat
				pre_line_right = x
	if left_line is not None:
		fill_points(left_line)
	if right_line is not None:
		fill_points(right_line)
	return left_line,right_line

cap = cv2.VideoCapture('outpy.avi')
if __name__ == '__main__':
	n=0
	while(cap.isOpened()):
		stime = time.time()
		ret, image_np = cap.read()
		#image_np=cv2.imread('difficult.png')
		res_binary,roi_binary,eyeBird_binary=get_processed_img(image_np)
		img_heigh = eyeBird_binary.shape[0]
		img_width = eyeBird_binary.shape[1]
		#lay tong cua nua phan duoi
		half_bottom_histogram = np.sum(eyeBird_binary[0:int(img_heigh/2),:], axis=0)

		#sub=np.sum(eyeBird_binary[int(img_heigh/16)*15:img_heigh,:], axis=0)
		#print (sub)
	
		right_x,_=find_first_point (arr=half_bottom_histogram, 
			startPos=160, 
			endPos=half_bottom_histogram.shape[0], left_or_right=1, 
			windowLength_line=7, windowLength_empty=20, step=2, 
			threshold_empty=15, threshold_line=50)
		left_x,_=find_first_point (arr=half_bottom_histogram, 
			startPos=0, 
			endPos=160, left_or_right=0, 
			windowLength_line=7, windowLength_empty=20, step=2, 
			threshold_empty=15, threshold_line=50)

		binary_img = np.dstack((eyeBird_binary, eyeBird_binary, eyeBird_binary))*255

		if right_x is not None:
			right_line = find_line (binary=eyeBird_binary,nWindows=16,first_point=right_x, 
							left_or_right=1, bottom_or_top=1)
			for point in right_line:
				cv2.circle(binary_img,(point[0],point[1]), 5, (0,255,0), -1)
		if left_x is not None:
			left_line = find_line (binary=eyeBird_binary,nWindows=16,first_point=left_x, 
							left_or_right=0, bottom_or_top=1)
			for point in left_line:
				cv2.circle(binary_img,(point[0],point[1]), 5, (255,0,0), -1)

		
		if right_x is not None:
			cv2.circle(binary_img,(int(right_x),200), 5, (0,0,255), -1)
		if left_x is not None:
			cv2.circle(binary_img,(int(left_x),200), 5, (0,0,255), -1)
		cv2.imshow('raw',image_np)
		cv2.imshow('binary',binary_img)
		print ('{} FPS'.format(1.0/(time.time() - stime)))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
