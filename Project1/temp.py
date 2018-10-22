# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def np_drop(feature, index, axis):
    feature_selected = np.delete(feature, index, axis = axis)
    
    return feature_selected


def np_count(feature_selected):
    feature_number = feature_selected.shape[0]
    
    return feature_number

def np_replace(feature_selected, to_be_replaced, replacement):
    feature_selected[np.where(feature_selected == to_be_replaced)] = replacement
    
    return feature_selected

def np_describe(feature_selected):
    describe = np.zeros([8, feature_selected.shape[1]])
    describe[0, :] = np.sum(np.ones_like(feature_selected), axis = 0)
    describe[1, :] = np.mean(feature_selected, axis = 0)
    describe[2, :] = np.std(feature_selected, axis = 0)
    describe[3, :] = np.min(feature_selected, axis = 0)
    describe[4, :] = np.percentile(feature_selected, 25, axis=0)
    describe[5, :] = np.percentile(feature_selected, 50, axis=0)
    describe[6, :] = np.percentile(feature_selected, 75, axis=0)
    describe[7, :] = np.max(feature_selected, axis=0)
    
    return describe

def 

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
rr = np_drop(arr, 1, 1)

