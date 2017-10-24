"""
Created on Sat Oct 14 08:27:19 2017

@author: zhi
"""
import numpy as np
import time
import cv2

# Image preprocessing
def image_preprocessing(raw_image):
    start_time = time.time()
    # Convert to numpy array
    image = np.array(raw_image)
    # uint8->double
    if image.dtype=='uint8':
        image = np.float32(image)/255.0
    # Saturate
    image[image<0.1] = 0.1
    print("---image_preprocessing costs %s seconds ---" % (time.time() - start_time))
    return image

# Transform color space
def color_channel_extraction(image):
    start_time = time.time()
    #v_channel = np.max(image,axis=2)
    #s_channel = (v_channel-np.min(image,axis=2))/v_channel
    HSV = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    s_channel = HSV[:,:,1]
    v_channel = HSV[:,:,2]
    r_channel = image[:,:,0]
    g_channel = image[:,:,1]
    b_channel = image[:,:,2]
    print("---color_channel_extraction costs %s seconds ---" % (time.time() - start_time))
    return v_channel, s_channel, r_channel, g_channel, b_channel

# Apply color filter on the image
def weighted_channels(image):
    v_channel, s_channel, r_channel, g_channel, b_channel = color_channel_extraction(image)
    start_time = time.time()
    weight_v = np.clip(v_channel+v_channel-1,0,1)
    weight_s = np.clip(s_channel+s_channel-1,0,1)
    #weight_all = weight_v*weight_s
    #weighted_r = weight_all*r_channel
    #weighted_g = weight_all*g_channel
    #weighted_b = weight_all*b_channel
    weight_all = cv2.multiply(weight_v,weight_s)
    weighted_r = cv2.multiply(weight_all,r_channel)
    weighted_g = cv2.multiply(weight_all,g_channel)
    weighted_b = cv2.multiply(weight_all,b_channel)
    print("---weighted_channels costs %s seconds ---" % (time.time() - start_time))
    return weighted_r,weighted_g,weighted_b

# color space segmentation
def color_space_segmentation(image):
    weighted_r,weighted_g,weighted_b = weighted_channels(image)
    start_time = time.time()
    # set partition
    partition_r = weighted_r-weighted_g-weighted_g>0
    partition_g = weighted_g-weighted_r-weighted_r>0
    partition_y = np.logical_and(weighted_r-weighted_g-weighted_g<0, weighted_g-weighted_r-weighted_r<0)
    # segment pixel values to different partition
    segment_r = np.zeros(weighted_r.shape)
    segment_g = np.zeros(weighted_r.shape)
    segment_y = np.zeros(weighted_r.shape)
    segment_r[partition_r] = weighted_r[partition_r]
    segment_g[partition_g] = weighted_g[partition_g]
    temp1 = segment_y
    temp2 = segment_y
    temp1[partition_y] = weighted_r[partition_y]
    temp2[partition_y] = weighted_g[partition_y]
    # segment_y = np.max(np.stack([temp1,temp2],axis=2),axis=2)
    segment_y = cv2.max(temp1,temp2)
    # Threashold
    segment_r = np.clip(segment_r+segment_r-1,0,1)
    segment_g = np.clip(segment_g+segment_g-1,0,1)
    segment_y = np.clip(segment_y+segment_y-1,0,1)
    print("---color_space_segmentation costs %s seconds ---" % (time.time() - start_time))
    return segment_r,segment_g,segment_y

# Simple classifier
def simple_detector(raw_image,show_image=0):
    image = image_preprocessing(raw_image)
    segment_r,segment_g,segment_y = color_space_segmentation(image)
    start_time = time.time()
    score_r = np.sum(segment_r)
    score_g = np.sum(segment_g)
    score_y = np.sum(segment_y)
    print("---simple_detector costs %s seconds ---" % (time.time() - start_time))
    if np.max([score_r,score_g,score_y]) <= 0:
        return -1
    return np.argmax([score_r,score_y,score_g])

def simple_detector_ROSdebug(raw_image,show_image=0):
    image = image_preprocessing(raw_image)
    segment_r,segment_g,segment_y = color_space_segmentation(image)
    score_r = np.sum(segment_r)
    score_g = np.sum(segment_g)
    score_y = np.sum(segment_y)
    return [score_r,score_y,score_g]

    
