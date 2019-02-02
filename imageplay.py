import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from auto_canny import auto_canny
#load in dataset
imageset = []
cannys = []
for file in os.listdir("dataset"):
    if file.endswith(".png"):
        imageset.append(cv2.imread(os.path.join("dataset",file)))

for im in imageset:
    blurd = cv2.GaussianBlur(im, (0, 0), 3)
    edgyIm = cv2.addWeighted(im, 1.5, blurd,-0.5, 0)

    #unsharpened original image
    origcan = cv2.Canny(im,80,200)

    #sharpened image
    can = cv2.Canny(edgyIm,80,200)

    cannys.append(can)
    #canny output for original image, converted to RED
    origcan3 = cv2.cvtColor(origcan,cv2.COLOR_GRAY2RGB)
    origcan3[np.where((origcan3==[255,255,255]).all(axis=2))] = [0,0,255]

    #canny output for sharpened image, converted to RED
    can3 = cv2.cvtColor(can,cv2.COLOR_GRAY2RGB)
    can3[np.where((can3==[255,255,255]).all(axis=2))] = [0,0,255]

    #autocanny output for original, unsharpened image, converted to RED
    autocanorig = cv2.cvtColor(auto_canny(im),cv2.COLOR_GRAY2RGB)
    autocanorig[np.where((autocanorig==[255,255,255]).all(axis=2))] = [0,0,255]

    #autocanny output for sharpened,edgy image, converted to RED
    autocan = cv2.cvtColor(auto_canny(edgyIm),cv2.COLOR_GRAY2RGB)
    autocan[np.where((autocan==[255,255,255]).all(axis=2))] = [0,0,255]

    # newIm = cv2.add(im,origcan3)
    # edgynewIm = cv2.add(im,can3)
    # AnewIm = cv2.add(im,autocanorig)
    # AedgynewIm = cv2.add(im,autocan)

    newIm = origcan3
    edgynewIm = can3
    AnewIm = autocanorig
    AedgynewIm = autocan
    #
    # cnts = cv2.findContours(can, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cnts))
    # for i, cnt in enumerate(cnts):
    #     cnts[i] = cv2.convexHull(cnts[i])
    # cv2.drawContours(origcan3,cnts, 0, (0,255,0), 3)
    #
    # newIm = origcan3
    # edgynewIm = can3
    # AnewIm = autocanorig
    # AedgynewIm = autocan

    # print(im.shape)
    # print(can.shape)
    stack = np.hstack((np.vstack((cv2.resize(newIm, (0,0), fx=0.5, fy=0.5),cv2.resize(edgynewIm, (0,0), fx=0.5, fy=0.5)) ),
                    np.vstack((cv2.resize(AnewIm, (0,0), fx=0.5, fy=0.5),cv2.resize(AedgynewIm, (0,0), fx=0.5, fy=0.5)) )))
    cv2.imshow('Image and Canny',stack)
    cv2.waitKey()
    cv2.destroyAllWindows()
