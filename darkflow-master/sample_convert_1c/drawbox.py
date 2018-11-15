import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector

# global constants
img = None
tl_list = []
br_list = []
object_list = []

# constants
image_folder = '/home/lee/Project/autocar/detectOject/BelgiumTSC_Training/Training/00022'
savedir = 'ban-sign'
obj = 'ban-sign'

def line_select_callback(clk):
	print (clk.xdata, clk.ydata)

if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1)
        image = cv2.imread(image_file.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        print (image.shape[1],image.shape[0])
        plt.connect('button_press_event', line_select_callback)
        plt.show()