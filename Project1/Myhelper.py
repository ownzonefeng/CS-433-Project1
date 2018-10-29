#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 23:20:49 2018

@author: wentao
"""
import numpy as np

"""generate test set and train set"""
def cross_validation_set(y, x, k_indices, k):
    x_k_test = x[k_indices[k],:]
    y_k_test = y[k_indices[k]]
    x_train = np.delete(x, k_indices[k], 0)
    y_train = np.delete(y, k_indices[k], 0)
    return y_train, x_train, y_k_test, x_k_test
    
"""build k indices for k-fold."""
def build_k_indices(y, k_fold):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

"""calculate the percentage of right prediction"""
def calculate_right_rate(original_label, predict_label):
    diff = abs(original_label - predict_label)# the sum of conflict labels
    rate = np.sum(diff / np.max(diff)) / original_label.shape[0]
    return 1 - rate

"""turn the labels into binary"""
def binary_label(y_pred):
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

"""log-scale and standard normalisation of features"""
def log_normal(array):
    for column_index in range(array.shape[1]):
        if np.all(array[:, column_index] >= 0):
            array[:, column_index] = np.log(array[:, column_index] + 0.0001)
        else:
            mean = np.mean(array[:, column_index])
            std = np.std(array[:, column_index])
            array[:, column_index] = (array[:, column_index] - mean) / std
    return array

"""Calculate the mse"""
def calculate_mse(e):
    return 1/2*np.mean(e**2)


"""Calculate the mae"""
def calculate_mae(e):
    return np.mean(np.abs(e))

"""Calculate the mse using observed value, features, and models"""
def compute_loss(y, tx, w):
    L = 0.5 * np.mean((y - tx @ w) ** 2, axis = 0)
    return L

"""build the polynomial feature"""
def build_poly_feature(array):
    dim_col = array.shape[1]
    feature_num =  1 + dim_col + dim_col + calculate_combination(dim_col, 2) + dim_col ** 2 + calculate_combination(dim_col,3)
    array_poly = np.ones((array.shape[0], int(feature_num)))
    array_poly[:, 1:dim_col + 1] = array
    col_loc = dim_col + 1
    
    # calculate the product of different feature with the order = 2
    # like a^2, b^2, a * b
    for i in range(array.shape[1]):
        for j in range(array.shape[1])[i:]:
            array_poly[:, col_loc] = array[:, i] * array[:, j]
            col_loc += 1
            
    # calculate the product of different feature with the order = 3
    # like a^2b, ab^2
    for i in range(array.shape[1]):
        for j in range(array.shape[1]):
            array_poly[:, col_loc] = (array[:, i] ** 2) * array[:, j]
            col_loc += 1
        
    # calculate the product of different feature with the order = 3
    # like abc, bcd, cde
    for i in range(array.shape[1]):
        for j in range(array.shape[1])[i + 1:]:
            for k in range(array.shape[1])[j + 1:]:
                array_poly[:, col_loc]  = array[:, i] * array[:, j] * array[:, k]
                col_loc += 1
                
    return array_poly

"""calculate the number of results when choosing y items from x items"""
def calculate_combination(x, y):
    comb = np.math.factorial(x) / np.math.factorial(y) / np.math.factorial(x - y) 
    return comb
            
    
    
