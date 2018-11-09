import numpy as np

def sum_window (array, startPos=0, length=1):
	ret=0
	for i in range (length):
		ret=ret+array[i+startPos]
	return ret

#histogram: bieu do da duoc phan tich sau khi chia thanh cac manh
#windowLength: Do dai cua so de tinh sum
#emptyLength: Do dai khoang cach (cho phep) cua 2 duong line
#thresholdEmpty: nguong cho phep de nhan do la ko line
#thresholdSum: nguong cho phep de nhan do la line
#ratio: duong phan biet line trai phai
#   +mau ratio cang lon -> vung line trai cang hep, line phai lon
#   +mau ratio cang nho -> vung line trai lon, line phai nho
def analyze_histogram (histogram,windowLength=1,emptyLength=1,thresholdEmpty=0,thresholdSum=3,ratio=0.5):
	histogramLeng=histogram.shape[0]
	maxSumIndex=-1
	maxSum=0
	preSumWindow=sum_window(histogram,0,windowLength)
	if preSumWindow>=thresholdSum:
		maxSumIndex=0
		maxSum=preSumWindow
	for iLeft in range(1,histogramLeng-windowLength+1):
		emptyFlag=True
		sumWindow = preSumWindow + histogram[iLeft+windowLength-1] - histogram[iLeft-1] #cong dau tru cuoi
		preSumWindow=sumWindow
		if sumWindow>maxSum and sumWindow>=thresholdSum:
			maxSum=sumWindow
			maxSumIndex=iLeft
			for iEmpty in range(emptyLength): #xuat hien vung nghi van line
				iHistogram=maxSumIndex+windowLength+iEmpty
				if  iHistogram< histogramLeng:
					if histogram[iHistogram] > thresholdEmpty:
						emptyFlag=False
						break
				else:
					emptyFlag=False
					break
		else:
			emptyFlag=False
		if emptyFlag:
			break

	ret = np.array([maxSumIndex,-1])

	if maxSumIndex != -1: #neu phat hien thay 1 line
		startPos = maxSumIndex + windowLength + emptyLength#new start pos
		maxSumIndex=-1
		if startPos < histogramLeng: #chua cham toi cuoi cung cua mang
			maxSum=0
			preSumWindow=sum_window(histogram,startPos,windowLength)
			if preSumWindow>=thresholdSum:
				maxSumIndex=startPos
				maxSum=preSumWindow
			for iRight in range(startPos+1,histogramLeng-windowLength+1):
				emptyFlag=True
				if iRight+windowLength-1 >= histogramLeng:
					break #vuot qua do dai mang
				sumWindow = preSumWindow + histogram[iRight+windowLength-1] - histogram[iRight-1] #cong dau tru cuoi
				preSumWindow=sumWindow
				if sumWindow>maxSum and sumWindow>=thresholdSum:
					maxSum=sumWindow
					maxSumIndex=iRight
					for iEmpty in range(emptyLength): #xuat hien vung nghi van line
						iHistogram=maxSumIndex+windowLength+iEmpty
						if  iHistogram< histogramLeng:
							if histogram[iHistogram] > thresholdEmpty:
								emptyFlag=False
						else:
							emptyFlag=False
				else:
					emptyFlag=False
				if emptyFlag:
					break
	ret[1]=maxSumIndex
	if ret[0] != -1:
		if ret[1] == -1:
			if ret[0] > float(histogramLeng)*ratio:
				ret[1]=ret[0]
				ret[0]=-1
	return ret

#thresholdGapLine: do chenh lech cho phep giua 2 pice co line
def rearrange_analyze_histogram (analyzed_histogram,slidingPice,thresholdGapLine=7):
	leftTotal=0
	rightTotal=0
	countAverage=0
	countDetectedLeft=0
	countDetectedRight=0
	leftAverage = 0
	rightAverage = 0
	for iSlide in range(slidingPice):
		if analyzed_histogram[iSlide][0] != -1 and analyzed_histogram[iSlide][1] != -1: #ca 2 ben deu phat hien
			leftTotal += analyzed_histogram[iSlide][0]
			rightTotal += analyzed_histogram[iSlide][1]
			countAverage += 1
	if countAverage > 0:
		leftAverage = float(leftTotal)/countAverage
		rightAverage = float(rightTotal)/countAverage

	for iSlide in range(slidingPice):
		if analyzed_histogram[iSlide][0] != -1 and analyzed_histogram[iSlide][1] == -1: #nham line phai thanh trai
			if abs(rightAverage - analyzed_histogram[iSlide][0]) <= thresholdGapLine:
				analyzed_histogram[iSlide][1] = analyzed_histogram[iSlide][0]
				analyzed_histogram[iSlide][0] = -1
		elif analyzed_histogram[iSlide][1] != -1 and analyzed_histogram[iSlide][0] == -1: #nham line trai thanh phai
			if abs(leftAverage - analyzed_histogram[iSlide][1]) <= thresholdGapLine:
				analyzed_histogram[iSlide][0] = analyzed_histogram[iSlide][1]
				analyzed_histogram[iSlide][1] = -1
		elif analyzed_histogram[iSlide][0] != -1 and analyzed_histogram[iSlide][1] != -1: #loc nhieu cho 2 line
			if abs(leftAverage - analyzed_histogram[iSlide][0]) > thresholdGapLine:
				analyzed_histogram[iSlide][0] = -1
			if abs(rightAverage - analyzed_histogram[iSlide][1]) > thresholdGapLine:
				analyzed_histogram[iSlide][1] = -1
		if analyzed_histogram[iSlide][0] != -1:
			countDetectedLeft += 1
		if analyzed_histogram[iSlide][1] != -1:
			countDetectedRight += 1
	return analyzed_histogram, countDetectedLeft, countDetectedRight

def list_points (analyzed_histogram,slidingPice,piceLength):
	leftPoints = rightPoints = None
	for iSlide in range (slidingPice):
		if (analyzed_histogram[iSlide][0] != -1): #cac diem nam ben line trai
			if leftPoints is None:
				leftPoints = np.array([[analyzed_histogram[iSlide][0], (iSlide+1)*piceLength]])
			else:
				temp = np.array([[analyzed_histogram[iSlide][0], (iSlide+1)*piceLength]])
				leftPoints = np.append(leftPoints,temp,axis=0)
		if (analyzed_histogram[iSlide][1] != -1):
			if rightPoints is None:
				rightPoints = np.array([[analyzed_histogram[iSlide][1], (iSlide+1)*piceLength]])
			else:
				temp = np.array([[analyzed_histogram[iSlide][1], (iSlide+1)*piceLength]])
				rightPoints = np.append(rightPoints,temp,axis=0)
	return leftPoints,rightPoints

# threshold khong duoc qua 1
def suit_lines (leftPoints,rightPoints,thresholdParallel=2, thresholdDistance=16):
	ret = None
	if leftPoints is None or rightPoints is None:
		return None
	for iLeft in range(leftPoints.shape[0]-1):
		for iRight in range(rightPoints.shape[0]-1):
			if (abs((leftPoints[iLeft][0]-leftPoints[iLeft+1][0]) - (rightPoints[iRight][0]-rightPoints[iRight+1][0])) <= thresholdParallel and #|(x1-x2) - (x3-x4)| <= thresholdParallel => song song
			   ((abs(leftPoints[iLeft][1] - rightPoints[iRight][1]) <= thresholdDistance and
				abs(leftPoints[iLeft][1] - rightPoints[iRight+1][1]) <= thresholdDistance) or
				(abs(leftPoints[iLeft+1][1] - rightPoints[iRight][1]) <= thresholdDistance and
				abs(leftPoints[iLeft+1][1] - rightPoints[iRight+1][1]) <= thresholdDistance))):
				if ret is None:
					ret = np.array([[[[leftPoints[iLeft][0],leftPoints[iLeft][1]],[leftPoints[iLeft+1][0],leftPoints[iLeft+1][1]]],[[rightPoints[iRight][0],rightPoints[iRight][1]],[rightPoints[iRight+1][0],rightPoints[iRight+1][1]]]]])
				else:
					temp = np.array([[[[leftPoints[iLeft][0],leftPoints[iLeft][1]],[leftPoints[iLeft+1][0],leftPoints[iLeft+1][1]]],[[rightPoints[iRight][0],rightPoints[iRight][1]],[rightPoints[iRight+1][0],rightPoints[iRight+1][1]]]]])
					ret = np.append(ret,temp,axis=0)
	return ret
		

			
