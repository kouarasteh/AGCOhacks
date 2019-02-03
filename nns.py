import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import os
from auto_canny import auto_canny

#load in dataset
imageset = []
cannys = []
for file in os.listdir("dataset"):
    if file.endswith(".png"):
        imageset.append(cv2.imread(os.path.join("dataset",file)))

def boost_green(inputImage, booster):
    image = inputImage
    for i in range(len(image)):
        for j in range(len(image[0])):
            if (np.argmax(image[i][j]) == 1):
                image[i][j][1] *= booster
    return image

def get_kmeans_labels(image, K, random_state=0):
    ''' Function to get the K-means labels
        for each pixel in the image
        Args:
            image: [ndarray (M x N x n_channels)] RGB image
            K: [int] number of clusters
        Returns:
            labels: [ndarray (M x N)] label image same
                    width and height as original image
        Hint:
            - Use the KMeans function from sklearn already imported
            - set the number of clusters and random state
            - reshape image array to a img_shape[0]*img_shape[1] x 3 array
              before feeding it to the KMeans
            - remember to reshape the output labels for KMeans.fit()
            - Make sure to try different random_state parameters for KMeans
               and choose the one that you think works best.
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height = len(image)
    width = len(image[0])
    pixs = [image[i][j] for i in range(height) for j in range(width)]
    #image[y][x] = pixs[540*y + x]
    kclass = KMeans(n_clusters=K,random_state=0)
    kclass.fit(pixs)
    labs = kclass.labels_
    truelabels = np.empty((height,width))
    for i in range(height):
        for j in range(width):
            truelabels[i][j] = labs[width*i + j]
    return truelabels

def get_slic_segmentation(image, num_segments):
    ''' Function to get the K-means labels for each pixel in the image
        Args:
            image: [ndarray (M x N x n_channels)] RGB image
            num_segments: [int] number of segments desired
        Returns:
            labels: [ndarray (M x N)] label image same
                    width and height as original image
        Hint:
            - Use the slic() function from skimage.segmentation
            - Pass the image and num_segments
            - You can vary the 'compactness' factor to vary the
              weight assigned to the x-y coordinates
    '''
    labels = slic(image,n_segments=num_segments,compactness=10)
    return labels

def colorzones(image,labels,toplabel):
    mask = np.zeros_like(image)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if labels[i][j] == toplabel:
                mask[i][j] = 255
    return mask

def getMaskedImage(image):
    #blur image
    blur = cv2.bilateralFilter(image,9,250,250)
    #kmeans image
    kmeans_8_labels  = get_kmeans_labels(blur,  8, 0)
    unique, counts = np.unique(kmeans_8_labels, return_counts=True)
    top2 = np.argsort(counts)[-3:]
    kmeans_8_labels[kmeans_8_labels == top2[1]] = top2[0]
    kmeans_8_labels[kmeans_8_labels == top2[2]] = top2[0]
    mask = colorzones(img,kmeans_8_labels,top2[0])
    mask[np.where((mask==255).all(axis=2))] = [0,255,0]
    final = cv2.addWeighted(image, 1.0, mask,0.25, 0)
    return final

for image in imageset:
    img = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    m = getMaskedImage(img)
    cv2.imshow('Image and Mask',np.vstack((img,m)))
    cv2.waitKey()

cv2.destroyAllWindows()
