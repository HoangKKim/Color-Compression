import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import random

def initCentroids(img_1d, k_clusters, init_centroids):
    if(init_centroids == 'random'):
        centroids = [np.random.choice(256, size = (img_1d.shape[1]), replace = False) for _ in range(k_clusters)]    
    elif(init_centroids == 'in_pixels'):
        uniqPixels = np.unique(img_1d, axis=0)
        if(len(uniqPixels) < k_clusters):
            k_clusters = len(uniqPixels)
        index = np.random.choice(len(uniqPixels), size = k_clusters, replace = False)
        centroids = [img_1d[i] for i in index]
    return centroids

def labelPixels(img_1d, k_clusters, centroids, labelArr, clusterList):
    for i in range(len(img_1d)):
        distance = np.linalg.norm(centroids - img_1d[i], axis=1)
        labelArr[i] = distance.argmin()
        clusterList[labelArr[i]].append(i)
    return labelArr, clusterList

def updateCentroids(img_1d, centroids, k_clusters, clusterList):
    for i in range(k_clusters):
        if(clusterList[i]):
            centroids[i] = np.mean([img_1d[j] for j in clusterList[i]], axis=0)
    return centroids

def kmeans(img_1d, k_clusters, max_iter, init_centroids='random'):       
    centroids = initCentroids(img_1d, k_clusters, init_centroids)
    labelArr = [None for _ in range(len(img_1d))]
    
    while(max_iter):
        preLabels = labelArr.copy()
        max_iter -=1   
        clusterList = [[] for _ in range(k_clusters)]

        labelArr, clusterList = labelPixels(img_1d, k_clusters, centroids, labelArr, clusterList)
                    
        if(labelArr == preLabels):
            break
   
        centroids = updateCentroids(img_1d, centroids, k_clusters, clusterList)

    return np.array(centroids).astype(int), np.array(labelArr)

def colorCompression(imgName, k_cluster, max_iter, init_centroids, extension):
    img = Image.open(imgName, mode="r")
    img = img.convert('RGB')
    img = np.array(img)
    img_1d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    centroids, labels = kmeans(img_1d, k_cluster, max_iter, init_centroids)

    for i in range(len(img_1d)):
        img_1d[i] = centroids[labels[i]]
    img_1d = img_1d.reshape(img.shape).astype(np.uint8)
    
    fileName = f"{imgName.split('.')[0]}_k{k_cluster}_{init_centroids}.{extension}"
    Image.fromarray(img_1d, 'RGB').save(fileName)
    
if __name__ == '__main__':
    while True:
        imgName = input('Name of image: ')
        if(not os.path.isfile(imgName)):
            print("Error! File not found.")
        else: break
    while True: 
        extension = input('Save format (png/pdf): ')
        if(extension not in ['png', 'pdf']):
            print('Error! Invalid save format')
        else: break
    
    colorCompression(imgName, 2, 10, 'in_pixels', extension)
    