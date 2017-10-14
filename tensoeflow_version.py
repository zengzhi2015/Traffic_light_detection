#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:27:19 2017

@author: zhi
"""

#import tensorflow as tf
import numpy as np

# Do not inclue the following when in ROS projects
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from matplotlib.colors import rgb_to_hsv

##%% Define the Holo kernal
#kernal_hole = np.array([[ 0.0, 0.0,-1.0,-1.0,-1.0,-1.0, 0.0, 0.0],
#                        [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0],
#                        [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,-1.0],
#                        [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,-1.0],
#                        [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,-1.0],
#                        [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,-1.0],
#                        [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0],
#                        [ 0.0, 0.0,-1.0,-1.0,-1.0,-1.0, 0.0, 0.0]])
## normalize the positive entries
#kernal_hole[kernal_hole>0] /= np.sum(kernal_hole[kernal_hole>0]);
## normalize and PENALIZE the negative entries !!!!!!!!!!!!!
#kernal_hole[kernal_hole<0] /= -0.66*np.sum(kernal_hole[kernal_hole<0]);
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
def color_channel_extraction(image):
    v_channel = np.max(image,axis=2)
    s_channel = (v_channel-np.min(image,axis=2))/v_channel
    r_channel = image[:,:,0]
    g_channel = image[:,:,1]
    b_channel = image[:,:,2]
    return v_channel, s_channel, r_channel, g_channel, b_channel

#%% Apply color filter on the image
def weighted_channels(image):
    v_channel, s_channel, r_channel, g_channel, b_channel = color_channel_extraction(image)
    weight_v = np.clip(v_channel*2-1,0,1)
    weight_s = np.clip(s_channel*2-1,0,1)
    weighted_r = weight_v*weight_s*r_channel
    weighted_g = weight_v*weight_s*g_channel
    weighted_b = weight_v*weight_s*b_channel
#    weighted_image = np.stack([weighted_r,weighted_g,weighted_b],axis=2)
#    fig = plt.figure(figsize=(10,10))
#    fig.add_subplot(2,2,1)
#    plt.imshow(weighted_image, cmap='gray')
#    fig.add_subplot(2,2,2)
#    plt.imshow(weighted_r, cmap='gray')
#    fig.add_subplot(2,2,3)
#    plt.imshow(weighted_g, cmap='gray')
#    fig.add_subplot(2,2,4)
#    plt.imshow(weighted_b, cmap='gray')
#    fig.tight_layout()
    return weighted_r,weighted_g,weighted_b

#%%
#%% Simple classifier
def simple_detector(image):
    filtered_r,filtered_g,filtered_b = weighted_channels(image)
#    fig = plt.figure(figsize=(10,10))
#    fig.add_subplot(2,2,1)
#    plt.imshow(image)
#    fig.add_subplot(2,2,2)
#    plt.imshow(filtered_r)
#    fig.add_subplot(2,2,3)
#    plt.imshow(filtered_g)
#    fig.add_subplot(2,2,4)
#    plt.imshow(filtered_b)
#    fig.tight_layout()
    score_r = np.sum(filtered_r)
    score_g = np.sum(filtered_g)
    score_b = np.sum(filtered_b)
    return np.argmax([score_r,score_g,score_b])
#%% Define the computation graph


#%% Overall test
if __name__ == '__main__':
    # Image reading and preprocessing
    raw_image = mpimg.imread('dayClip10--00023.png')
    # raw_image = mpimg.imread('dayClip5--00015.png')
    # raw_image = mpimg.imread('dayClip5--01177.png')
    # raw_image = mpimg.imread('img_10_575.png')
    # raw_image = mpimg.imread('img_12_388.png')
    image = image_preprocessing(raw_image)
    result = simple_detector(image)
    print('The detection result is {}.'.format(result))
    