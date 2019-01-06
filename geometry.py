import numpy as np
import math

#tim cac cap duong thang gan nhu song song o gan nhau 1 khoang khong qua thresholdDistance
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

def list_mid_points (suitPoint):
	if suitPoint is None:
		return None
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

def average_mid_point (midPoints):
	if midPoints is None:
		return None
	TotalX=0
	TotalY=0
	for iMidPoints in range(midPoints.shape[0]):
		TotalX += midPoints[iMidPoints][0]
		TotalY += midPoints[iMidPoints][1]
	return np.array([TotalX/midPoints.shape[0],TotalY/midPoints.shape[0]])

def caculate_angle_speed (ImgShapeX,ImgShapeY,midPoint,ratioAngle,maxSpeed,ratioSpeed):
	AB=abs(ImgShapeY-midPoint[1])
	AC=abs(ImgShapeX/2 - midPoint[0])
	BC=math.sqrt(float(AB*AB) + AC*AC)
	angle=0
	if midPoint[0] > ImgShapeX/2:
		angle=1 #be ben phai
	elif midPoint[0] < ImgShapeX/2:
		angle=-1 #be ben trai
	else:
		angle=0
		speed=maxSpeed*ratioSpeed
		return angle,speed
	angle = angle * np.arctan(float(AC)/AB)/np.pi*180 * ratioAngle * (float(midPoint[1])/ImgShapeY)
	if abs(angle) < 1:
		speed=maxSpeed * ratioSpeed
	elif abs(angle) < 2:
		speed=maxSpeed * 0.98 * ratioSpeed
	elif abs(angle) < 3:
		speed=maxSpeed * 0.95 * ratioSpeed
	elif abs(angle) < 4:
		speed=maxSpeed * 0.92 * ratioSpeed
	elif abs(angle) < 5:
		speed=maxSpeed * 0.9 * ratioSpeed
	elif abs(angle) < 6:
		speed=maxSpeed * 0.88 * ratioSpeed
	elif abs(angle) < 10:
		speed=maxSpeed * 0.8 * ratioSpeed
	else:
		speed=maxSpeed * 0.5 * ratioSpeed
	return angle,speed

#kiem tra diem co nam trong tam giac hay ko
def point_in_triangle (pt, v1, v2, v3):
	b1=sign(pt,v1,v2)<0
	b2=sign(pt,v2,v3)<0
	b3=sign(pt,v3,v1)<0

	return ((b1==b2) and (b2==b3))

def sign (p1, p2, p3):
	#(p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
	return (float((p1['x']-p3['x'])*(p2['y']-p3['y'])) - float((p2['x']-p3['x'])*(p1['y']-p3['y'])))

#return 0: nam ben trai
#return 1: nam o giua
#return 2: nam ben phai
#return 3: khong nam trong vung roi


#     L *A*********************B* M
#       *  *                 *  *
#       * *                   * *
#       **                     **
#     F *                       * C
#       *                       *
#     E ************************* D


#      G ******** H
#        *      *
#      I ******** K
def rectangle_in_roi (rec):
	#rec['topleft']['x']
	#rec['topleft']['y']
	#rec['bottomright']['x']
	#rec['bottomright']['y']

	#rectangle
	G={
		'x':rec['topleft']['x'],
		'y':rec['topleft']['y']
	}
	H={
		'x':rec['bottomright']['x'],
		'y':rec['topleft']['y']
	}
	K={
		'x':rec['bottomright']['x'],
		'y':rec['bottomright']['y']
	}
	I={
		'x':rec['topleft']['x'],
		'y':rec['bottomright']['y']
	}

	#ROI
	A={
		'x':50,
		'y':70
	}
	B={
		'x':270,
		'y':70
	}
	C={
		'x':320,
		'y':160
	}
	D={
		'x':320,
		'y':240
	}
	E={
		'x':0,
		'y':240
	}
	F={
		'x':0,
		'y':160
	}
	L={
		'x':0,
		'y':100
	}
	M={
		'x':320,
		'y':100
	}

	#vung tren
	if I['y'] < A['y']:
		return 3
	#vung nam trong tam giac ngoai
	if point_in_triangle (I,B,M,C):
		return 3
	if point_in_triangle (K,L,A,F):
		return 3
	#xac dinh diem x giua hinh chu nhat
	Xmid=abs(int((H['x']-G['x'])/2))+G['x']
	#neu hinh chu nhat nam trong vung mid 320/2=160
	if (abs(160-Xmid) <= 10):
		return 1
	#nam ben trai
	if (160-Xmid) > 0:
		return 0
	#nam ben phai
	if (160-Xmid) < 0:
		return 2

