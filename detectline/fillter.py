import cv2 
import numpy as np


def region_of_interest(img):

    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # Create a match color with the same color channel counts.
    match_mask_color = 255


    # Fill inside the polygon
    #phia tren
    lower_left=[0,160]
    lower_right=[320,160]
    top_left=[80,100]
    top_right=[240,100]

    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
      
    cv2.fillPoly(mask, vertices, match_mask_color)

    #phia duoi
    lower_left=[0,240]
    lower_right=[320,240]
    top_left=[0,160]
    top_right=[320,160]

    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img.astype(int), mask.astype(int))

    return masked_image.astype(float)

def perspective_transform(img):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

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

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

    return warped, unwarped, m, m_inv

def get_processed_img (img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)

    minThreshold = np.array([0, 0, 180])
    maxThreshold = np.array([179, 30, 255])
    mask = cv2.inRange(hsv, minThreshold, maxThreshold)

    #cv2.imshow('mask',mask)

    minLaneInShadow = np.array([90, 43, 97]) 
    maxLaneInShadow = np.array([120, 100, 171]) 
    landShadow = cv2.inRange(hsv, minLaneInShadow, maxLaneInShadow)
    #cv2.imshow('landShadow',landShadow)

    res = np.bitwise_or(landShadow,mask)

    res_binary = np.ones_like(res).astype(np.uint8)
    res_binary = np.bitwise_and(res,res_binary)

    roi_binary = region_of_interest(res_binary)

    eyeBird_binary,_,_,_ = perspective_transform(roi_binary)

    return res_binary,roi_binary,eyeBird_binary

if __name__ == '__main__':
    image_np=cv2.imread('difficult.png')
    res_binary,roi_binary,eyeBird_binary=get_processed_img(image_np)
    binary_img=np.dstack((eyeBird_binary, eyeBird_binary, eyeBird_binary))*255
    cv2.imshow('bi',binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
