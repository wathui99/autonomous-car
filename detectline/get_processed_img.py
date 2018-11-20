import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

	
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


def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	"""
	Takes an image, gradient orientation, and threshold min/max values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Return the result
	return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	Return the direction of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output


def hls_thresh(img, thresh=(100, 255)):
	"""
	Convert RGB to HLS and threshold to binary image using S channel
	"""
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output


def combined_thresh(img):
	abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)
	mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
	dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
	hls_bin = hls_thresh(img, thresh=(170, 255))

	combined = np.zeros_like(dir_bin)
	combined[((abs_bin == 1) | ((mag_bin == 1) & (dir_bin == 1))) | (hls_bin == 1)] = 1

	return combined, abs_bin, mag_bin, dir_bin, hls_bin # DEBUG


def color_thr(s_img, v_img, s_threshold = (0,255), v_threshold = (0,255)):
    s_binary = np.zeros_like(s_img).astype(np.uint8)
    s_binary[(s_img > s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    v_binary = np.zeros_like(s_img).astype(np.uint8)
    v_binary[(v_img > v_threshold[0]) & (v_img <= v_threshold[1])] = 1
    col = ((s_binary == 1) | (v_binary == 1))
    return col

#the main thresholding operaion is performed here 
def thresholding(img, s_threshold_min = 113, s_threshold_max = 255, v_threshold_min = 234, v_threshold_max = 255,  k_size = 5):
    # Convert to HSV color space and separate the V channel
	imshape = img.shape
    #convert to HLS
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #read saturation channel
	s_channel = hls[:,:,2].astype(np.uint8)
    #Convert to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    #read the value channel
	v_channel = hsv[:,:,2].astype(np.uint8)
    #threshold value and saturation
	threshold_binary = color_thr(s_channel, v_channel, s_threshold=(s_threshold_min,s_threshold_max), v_threshold= (v_threshold_min,v_threshold_max)).astype(np.uint8)
	return threshold_binary

def process_img(img):
	combined, _, _, _, _ = combined_thresh(img)
	#combined = thresholding(img,230,240,220,255)
	#combined = thresholding(img,s_threshold_min = s_min, s_threshold_max = s_max, v_threshold_min = v_min, v_threshold_max = v_max,  k_size = 5)
	roi_binary=region_of_interest(combined)
	warped, unwarped, m, m_inv = perspective_transform(roi_binary)
	return combined, roi_binary, warped


if __name__ == '__main__':

	img = mpimg.imread('difficult.png')

	print (img)

	combined, _, _, _, _ = combined_thresh(img)

	print (combined[1])
	print(np.sum(combined[1:2,:], axis=0))

	roi_binary=region_of_interest(combined)

	warped, unwarped, m, m_inv = perspective_transform(roi_binary)

	plt.subplot(2, 3, 1)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)

	plt.subplot(2, 3, 2)
	plt.imshow(combined, cmap='gray', vmin=0, vmax=1)

	plt.subplot(2, 3, 3)
	plt.imshow(roi_binary, cmap='gray', vmin=0, vmax=1)

	plt.subplot(2, 3, 4)
	plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
	

	plt.tight_layout()
	plt.show()
