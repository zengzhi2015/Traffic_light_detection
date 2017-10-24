#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:21:02 2017

@author: zhi
"""

from simple_detector import simple_detector, simple_detector_ROSdebug
#import numpy as np
import pandas as pd
import matplotlib.image as mpimg
 #%%
if __name__ == '__main__':
    dataset_path = '/home/zhi/Downloads/data1/'
    dataset_info_path = '/home/zhi/Downloads/data1/img_test_data.csv'
    table = pd.read_csv(dataset_info_path)
    pos_count = 0
    neg_count = 0
    neg_index = []
    for image_name,value in zip(table.image_name,table.value):
        #print(image_name,value)
        image_path = dataset_path+image_name
        raw_image = mpimg.imread(image_path)
        result = simple_detector(raw_image)
        if result==value:
            pos_count+=1
        else:
            neg_index.append(image_path)
            neg_count+=1
    print(pos_count)
    print(neg_count)
    print(neg_index)

#%%
image_path = neg_index[0]
raw_image = mpimg.imread(image_path)
result = simple_detector(raw_image)
#%%
image_path = 'image.png'
raw_image = mpimg.imread(image_path)
result = simple_detector_ROSdebug(raw_image)
            