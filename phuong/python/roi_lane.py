import cv2
import numpy as np
# top right, bot righ, bot left, top left
def perspective_transform(img):
    imshape = img.shape
    #print (imshape)
    
    src = np.float32(
        [[0, 210],   #botton left
        [320, 210],  #botton right
        [122, 107],  #top left
        [200, 107]]) #top right
    dst = np.float32(
        [[100, 240],
        [220, 240],
        [100, 0],
        [220, 0]])
    #print ('dst : %s'%(dst))
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (imshape[1], imshape[0]) 
    perspective_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)    
    return perspective_img
def region_of_interest(img):
    #defining a blank mask to start with
    imshape = img.shape
    src = np.float32([[(130,80), \
                        (230,80), \
                        (319,239), \
                       (0,239)]])
    ignore_mask_color=255
    mask = np.zeros_like(img, dtype=np.uint8)
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    #print ('src ',vertices)    
    #cv2.namedWindow('mask', cv2.WINDOW_NORMAL)    
    cv2.fillPoly(mask, np.array([src], dtype=np.int32), ignore_mask_color)
    #cv2.imshow('mask',mask)    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def main():
    img = cv2.imread ("/home/hoaiphuong/Pictures/7.png")
    imshape = img.shape
    
    src = np.float32([[(0.5*imshape[1],0.416*imshape[0]), \
                        (0.64*imshape[1],0.416*imshape[0]), \
                        (0.937*imshape[1],imshape[0]), \
                       (0.218*imshape[1],imshape[0])]])
    #masked_image=region_of_interest(img, src)
    perspective_img=perspective_transform(img)
    #cv2.imshow('masked_image',masked_image)
    cv2.imshow('perspective_img',perspective_img)
    cv2.waitKey(0)  
#main()
