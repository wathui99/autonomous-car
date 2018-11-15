import numpy as np
import math

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
