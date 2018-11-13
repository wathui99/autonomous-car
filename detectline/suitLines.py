import numpy as np

def suit_lines (leftPoints,rightPoints,thresholdParallel=4, thresholdDistance=15):
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

def list_mid_points (suitPoint):
	midPoints=None
	for iSuitPoint in range(suitPoint.shape[0]):
		firstMidPoint=None
		secondMidPoint=None
		#M1
		x1=suitPoint[iSuitPoint][0][0][0]
		y1=suitPoint[iSuitPoint][0][0][1]
		#M2
		x2=suitPoint[iSuitPoint][0][1][0]
		y2=suitPoint[iSuitPoint][0][1][1]
		#M3
		x3=suitPoint[iSuitPoint][1][0][0]
		y3=suitPoint[iSuitPoint][1][0][1]
		#M4
		x4=suitPoint[iSuitPoint][1][1][0]
		y4=suitPoint[iSuitPoint][1][1][1]

		firstMidPoint = np.array([(x1+x3)/2,(y1+y3)/2])
		secondMidPoint = np.array([(x2+x4)/2,(y2+y4)/2])

		if midPoints is None:
			midPoints=np.array([[(firstMidPoint[0]+secondMidPoint[0])/2,(firstMidPoint[1]+secondMidPoint[1])/2]])
		else:
			temp=np.array([[(firstMidPoint[0]+secondMidPoint[0])/2,(firstMidPoint[1]+secondMidPoint[1])/2]])
			midPoints=np.append(midPoints,temp,axis=0)
	return midPoints

leftPoints = np.array([[135,90],[136,105],[137,120]])
rightPoints = np.array([[202,90],[204,105],[205,150]])

suitLines = np.array(
[[[[138,90],
   [137,105]],

  [[204,105],
   [204,120]]],


 [[[137,105],
   [137,120]],

  [[204,105],
   [204,120]]]])

midPoints = list_mid_points(suitLines)

print (midPoints)
