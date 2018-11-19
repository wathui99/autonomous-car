import numpy as np
import cv2
from get_processed_img import process_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

#tra ve 1 mang gom left line va right line
#[
#[[xleft,yleft],[xright,yright]], .... nWindows
#]
#Chu y: khong nhan dien duoc tra ve None
def detect_line(eyeBird_binary,nWindows=16):
	img_heigh = eyeBird_binary.shape[0]
	img_width = eyeBird_binary.shape[1]
	#lay tong cua nua phan duoi
	half_bottom_histogram = np.sum(eyeBird_binary[int(img_heigh/2):img_heigh,:], axis=0)
	#tim vi tri xuat hien cao nhat cua duong line
	max_sub_left,max_sub_pos_left = max_sub_array(half_bottom_histogram, 0, int(img_width/2), 10)
	max_sub_right,max_sub_pos_right = max_sub_array(half_bottom_histogram, int(img_width/2), int(img_width)-1, 10)
	#detect first point flag
	#neu khong tinh duoc thi se bo qua line ben do
	first_left_point = True
	first_right_point = True
	if (max_sub_left < 50): 
		first_left_point = False
	if (max_sub_right < 50):
		first_right_point = False
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
		print (sub_his)
		#tinh toa do line trai truoc
		if (first_left_point):
			x=find_pos_line(histogram=sub_his, windowLength=10, threshold_line=20, 
				threshold_empty=3, length_empty=50, count_empty=20, 
				pre_linePos=pre_line_left, max_2Lines=15)
			coo = np.array([[x,img_heigh - window_heigh*iWindow]])
			left_line = np.append(left_line,coo,axis=0) #them vao left line
			#cap nhat vi tri line moi
			if (x!=-1): #neu tim thay vi tri moi thi -> cap nhat
				pre_line_left = x
		#tinh toan toa do line phai
		if (first_right_point):
			x=find_pos_line(histogram=sub_his, windowLength=10, threshold_line=20, 
				threshold_empty=3, length_empty=50, count_empty=20, 
				pre_linePos=pre_line_right, max_2Lines=15)
			coo = np.array([[x,img_heigh - window_heigh*iWindow]])
			right_line = np.append(right_line,coo,axis=0) #them vao left line
			#cap nhat vi tri line moi
			if (x!=-1): #neu tim thay vi tri moi thi -> cap nhat
				pre_line_right = x
	return left_line,right_line


#tim vi tri cua duong line dua vao histogram
#windowLength la kich thuoc cua thanh truot de tinh tong (CHU Y TRUYEN SO LE)
#threshold_line la nguong tinh tong de nhan dien do la line
#threshold_empty la nguong tinh tong de nhan dien do la khong co line
#length_empty do dai cua ko co line
#count_empty so pixels de chap nhan do la ko line
#pre_linePos la vi tri cua line truoc do
#max_2Lines khoang cach lon nhat cua line hien tai va line truoc do
def find_pos_line(histogram, windowLength, threshold_line, threshold_empty, length_empty, count_empty, pre_linePos, max_2Lines):
	#do dai cua histogram
	his_shape=histogram.shape[0]
	#gia tri tong cua cac windows
	left_sum = 0
	right_sum = 0
	current_max = np.sum(histogram[pre_linePos-int(windowLength/2):pre_linePos+int(windowLength/2)+1])
	current_pos_max = pre_linePos
	#xuat phat tu pre_pos tim sub max tiep theo
	for iPos in range(1,max_2Lines+1):
		startPos = pre_linePos-int(windowLength/2)
		endPos = pre_linePos+int(windowLength/2)
		left_sum=np.sum(histogram[startPos-iPos:endPos+1-iPos])
		right_sum=np.sum(histogram[startPos+iPos:endPos+1+iPos])
		
		#print ('iPos=',iPos)
		#print ('startPos=',startPos)
		#print ('endPos=',endPos)
		#print ('left_sum=',left_sum)
		#print ('right_sum=',right_sum)

		#neu ben phai lon hon
		if (right_sum > left_sum):
			side_pos_max = pre_linePos + iPos
			side_max = right_sum
		else:
			side_pos_max = pre_linePos - iPos
			side_max = left_sum

		#neu gia tri tim duoc lon hon current
		if (side_max > current_max):
			current_max=side_max
			current_pos_max=side_pos_max

		#print ('current_max=',current_max)
		#print ('current_pos_max',current_pos_max)

	#neu tong thoa dieu kien cua threshold co line va xung quanh khong line
	not_empty=True
	good_rightSide=False
	good_leftSide=False
	count_empty_left=0
	count_empty_right=0
	for i in range(length_empty):

		if (current_pos_max - int(windowLength/2) -i) < 0: #cham bien ben trai
			good_leftSide=True
		if (current_pos_max + int(windowLength/2) +i) >= his_shape: #cham bien ben phai
			good_leftSide=True

		if histogram[current_pos_max - int(windowLength/2) -i] <= threshold_empty:
			count_empty_left+=1
		if histogram[current_pos_max + int(windowLength/2) +i] <= threshold_empty:
			count_empty_right+=1

	if count_empty_left >= count_empty:
		good_leftSide=True
	if count_empty_right >= count_empty:
		good_rightSide=True

	#print (current_max,count_empty_left,count_empty_right)

	if (current_max >= threshold_line) and good_rightSide and good_leftSide:
		return current_pos_max
	else:
		return -1


if __name__ == '__main__':

	arr = np.array([0,1,0,1,2,5,6,7,1,0,0,0,0,0,1,0,1,5,6,6,4,2,3,3,3])
	i = find_pos_line(histogram=arr, windowLength=3, threshold_line=10,
		threshold_empty=2, length_empty=5, count_empty=3, 
		pre_linePos=16, max_2Lines=3)

	print (i)

	bgr_img = cv2.imread('difficult.png')

	img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

	thres_binary, roi_binary, eyeBird_binary = process_img(img)

	left_line,right_line=detect_line(eyeBird_binary)
	print(left_line)
	print(right_line)

	plt.subplot(2, 3, 1)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)

	plt.subplot(2, 3, 2)
	plt.imshow(thres_binary, cmap='gray', vmin=0, vmax=1)

	plt.subplot(2, 3, 3)
	plt.imshow(roi_binary, cmap='gray', vmin=0, vmax=1)

	plt.subplot(2, 3, 4)
	plt.imshow(eyeBird_binary, cmap='gray', vmin=0, vmax=1)
		

	plt.tight_layout()
	plt.show()