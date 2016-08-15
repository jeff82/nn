# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:04:32 2016

@author: ZM
"""

# Harris Corner Detector
# Code - 4.18
# -*- coding:gb2312 -*-
#import cv2
import numpy as np
import numpy
import matplotlib.pyplot as plt
from skimage.feature import canny
from pic import *
from scipy.ndimage import filters

X_w=numpy.array([1,0,-1])
Y_w=numpy.array([1,0,-1])
def conv(dif,G,windowsz=3):
    '''
    2D assumed
    '''
    dx,dy=dif.shape
    dxg,dyg=G.shape
    sum=0
    
#    for x in range(windowsz):
#        for y in range(windowsz):
#            cv=numpy.dot(dif[x,y],G[x,y])
#            sum+=cv
    sum=numpy.dot(numpy.dot(G[0].reshape(1,windowsz),dif),G[1].reshape(windowsz,1))    
    return sum
# from scipy.ndimage import *
def Harris(img,op='H',w=4,k=0.1):
    # below is wrong for trying to create a
    # ndarray with demision number greater than 32
    # img = np.ndarray(img, dtype=np.float32)
    # if use np.mat, be ware of the difference in matrix multiply
    img = np.array(img,dtype=np.float32)
    # row,col = img.shape
    # out = np.zeros(img.shape)
    #difx,dify=np.gradient(img)

    # detect edge with cv2.Sobel
    # EdgeX = cv2.Sobel(img, cv2.CV_32FC1, 1, 0)
    # EdgeY = cv2.Sobel(img, cv2.CV_32FC1, 0, 1)
    # Mag = np.sqrt(EdgeX ** 2 + EdgeY ** 2)

    # or use canny edge detector from skimage.feature
    Mag = canny(img, 1, 0.4, 0.2)
#    plt.figure()
#    plt.imshow(Mag)
    G[0]=Y_w
    G[1]=X_w
    
    difx,dify=np.gradient(img)

    # compute autocorralation
    difx2=difx**2
    dify2=dify**2
    difxy=difx*dify

    # or use cv2.multiply
    # difx2 = cv2.multiply(difx,difx)
    # dify2 = cv2.multiply(dify,dify)
    # difxy = cv2.multiply(difx, dify)

    # mean filter in scipy.ndimage
    # A = uniform_filter(difx2,size=w)
    # B = uniform_filter(dify2,size=w)
    # C = uniform_filter(difxy,size=w)

    # or use mean filter in cv2
#    A = cv2.blur(difx2,(w,w))
#    B = cv2.blur(dify2,(w,w))
#    C = cv2.blur(difxy,(w,w))
    
    A = filters.gaussian_filter(difx2,5)
    B = filters.gaussian_filter(dify2,5)
    C = filters.gaussian_filter(difxy,5)

    if op =='H':
        out = A*B - C**2 -k*(A+B)**2
        out[Mag == 0] = 0
    else:
        out = (A*dify2-2*C*difxy+B*difx2)/(difx2+dify2+1)
        out[difx2+dify2==0]=0

    # next section for debug
    # plt.subplot(221);plt.imshow(img)
    # plt.subplot(222);plt.imshow(difx)
    # plt.subplot(223);plt.imshow(difx2);plt.show()
    # # plt.subplot(224);plt.imshow(C);plt.show()
    return out

if __name__ == '__main__':
#    image = cv2.imread('u.tif', flags=0)
#    image = cv2.resize(image, (100, 100))
    X,img=readimg()
    corimg = Harris(X[:,:,0], op='TI')
    #plt.subplot(121);
    plt.figure()
    plt.imshow(X[:,:,0]);plt.title('Original')
    spic=numpy.zeros((corimg.shape),dtype='uint8')
    #spic[xx,xx]=[[uint8(dd[0]),uint8(dd[1])] for dd,xx in corimg[:,:,0]] 
    spic=map(map, [numpy.int8, numpy.int8], corimg[:,:])
    img = Image.fromarray(numpy.array(spic))
   
    img.save(dir+"\ss"+".jpg")
    #plt.subplot(122);
    plt.figure()
    plt.imshow(corimg,);plt.title('Harris Corner')
    
    plt.figure()
    difx,dify=np.gradient(corimg)
    plt.imshow(difx)
    plt.show()