#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:30:26 2018

@author: wentao
"""

import datetime
from helpers import *
from implementations import *
import numpy as np
import matplotlib.pyplot as plt
height, weight, gender = load_data(sub_sample=False, add_outlier=False)
x, mean_x, std_x = standardize(height)
y, tx = build_model_data(x, weight)

loss_1, w_1 = least_squares_GD(y, tx, np.ones((tx.shape[1], 1)), 1000, 0.1)