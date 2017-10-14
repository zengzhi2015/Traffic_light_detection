#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:21:02 2017

@author: zhi
"""

from numpy_version import simple_detector
import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset_info_path = '/home/zhi/Downloads/data1/img_test_data.csv'
    table = pd.read_csv(dataset_info_path)