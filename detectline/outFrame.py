import cv2
import os

cap = cv2.VideoCapture('outpy.avi')
n=0;
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    cv2.imwrite('outFrame/{}.png'.format(n),frame)
    n+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()