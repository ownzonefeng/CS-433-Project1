#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:40:39 2018

@author: wentao
"""

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *

labels, features, ids = load_csv_data('train.csv') 
features_processed = np.delete(features, np.where(features[:, -1] == 0), axis = 0)
labels_processed = np.delete(labels, np.where(features[:, -1] == 0), axis = 0)
labels_processed = np.delete(labels_processed, np.where(features_processed[:, 28] == -999), axis = 0)
labels_processed = labels_processed.reshape(labels_processed.shape[0], 1)
features_processed = np.delete(features_processed, np.where(features_processed[:, 28] == -999), axis = 0)
features_processed = np.delete(features_processed, 22, axis = 1)
features_processed = features_processed[:, 13:]
#w = logistic_regression_ridge(labels, features_processed, max_iters = 100)