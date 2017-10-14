#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:27:19 2017

@author: zhi
"""

import tensorflow as tf
import numpy as np

# Do not inclue the following when in ROS projects
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import rgb_to_hsv

#%% Define the Holo kernal
kernal_hole = np.array([[ 0.0, 0.0,-1.0,-1.0,-1.0,-1.0, 0.0, 0.0],
                        [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0],
                        [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,-1.0],
                        [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,-1.0],
                        [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,-1.0],
                        [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,-1.0],
                        [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0],
                        [ 0.0, 0.0,-1.0,-1.0,-1.0,-1.0, 0.0, 0.0]])
# normalize the positive entries
kernal_hole[kernal_hole>0] /= np.sum(kernal_hole[kernal_hole>0]);
# normalize and PENALIZE the negative entries !!!!!!!!!!!!!
kernal_hole[kernal_hole<0] /= -0.66*np.sum(kernal_hole[kernal_hole<0]);
#%% Image preprocessing
def image_preprocessing(raw_image):
    # Convert to numpy array
    image = np.array(raw_image)
    # uint8->double
    if image.dtype=='uint8':
        image = np.double(image)
    # Saturate
    image[image<0.1] = 0.1
    return image
#%% Transform color space
def color_space_transform(image):
    pass
#%% Define the computation graph


#%% Overall test
if __name__ == '__main__':
    # Image reading and preprocessing
    #raw_image = mpimg.imread('dayClip10--00023.png')
    raw_image = mpimg.imread('dayClip5--01177.png')
    image = image_preprocessing(raw_image)
    hsv_image = rgb_to_hsv(image)
    fig = plt.figure(figsize=(16,16))
    fig.add_subplot(2,2,1)
    plt.imshow(image)
    fig.add_subplot(2,2,2)
#    plt.imshow(hsv_image[:,:,0])
#    temp_h = hsv_image[:,:,0]
#    temp_h -= 7.0/30.0
#    temp_h *= 5.0
#    temp_h[temp_h<0]=0.0
#    temp_h[temp_h>1]=1.0
#    plt.imshow(temp_h)
    # temp_s_weighted = image
    temp_s = hsv_image[:,:,1]
    temp_s = temp_s*2-1
    temp_s[temp_s<0]=0
    weight_s = np.stack([temp_s,temp_s,temp_s],axis=2)
    temp_v = hsv_image[:,:,2]
    temp_v = temp_v*2-1
    temp_v[temp_v<0]=0
    weight_v = np.stack([temp_v,temp_v,temp_v],axis=2)
    temp_s_weighted = image*weight_s*weight_v
    #temp_s=temp_s*temp_s
    # temp_s_weighted = temp_s_weighted*np.stack([hsv_image[:,:,1],hsv_image[:,:,1],hsv_image[:,:,1]],axis=2)
    # temp_s_weighted = temp_s_weighted*np.stack([temp_s,temp_s,temp_s],axis=2)
    plt.imshow(temp_s_weighted[:,:,1])
    fig.add_subplot(2,2,3)
    plt.imshow(hsv_image[:,:,1])
    # temp_s[temp_s<0.5]=0
    # plt.imshow(temp_s)
    fig.add_subplot(2,2,4)
    # plt.imshow(hsv_image[:,:,2])
    # temp_v[temp_v<0.5]=0
    plt.imshow(hsv_image[:,:,2])
    