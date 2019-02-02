import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#load in dataset
imageset = []
cannys = []
for file in os.listdir("dataset"):
    if file.endswith(".png"):
        imageset.append(cv2.imread(os.path.join("dataset",file)))

for im in imageset:
    can = cv2.Canny(im,100,200)
    cannys.append(can)
    print(can)
    can3 = cv2.cvtColor(can,cv2.COLOR_GRAY2RGB)
    # print(im.shape)
    # print(can.shape)
    # stack = np.vstack((im,can3))
    # cv2.imshow('Image and Canny',can)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


# image = cv2.imread('pinocchio.png')
# # I just resized the image to a quarter of its original size
# image = cv2.resize(image, (0, 0), None, .25, .25)
#
# grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# # Make the grey scale image have three channels
# grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
#
# numpy_vertical = np.vstack((image, grey_3_channel))
# numpy_horizontal = np.hstack((image, grey_3_channel))
#
# numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
# numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
#
# cv2.imshow('Main', image)
# cv2.imshow('Numpy Vertical', numpy_vertical)
# cv2.imshow('Numpy Horizontal', numpy_horizontal)
# cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
# cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
#
# cv2.waitKey()
