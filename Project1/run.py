#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 07:45:12 2018

@author: wentao
"""

import numpy as np
import matplotlib.pyplot as plt
from Myhelper import *
from proj1_helpers import *
from implementations import *

labels, features, ids = load_csv_data('train.csv') #load CSV file
labels = (labels + 1) /2 #transfer the label from [-1, 1] to [0, 1]

feature_dict = {"DER_mass_MMC":0,#mapping from feature name to feature index
"DER_mass_transverse_met_lep":1,
"DER_mass_vis":2,
"DER_pt_h":3,
"DER_deltaeta_jet_jet":4,
"DER_mass_jet_jet":5,
"DER_prodeta_jet_jet":6,
"DER_deltar_tau_lep":7,
"DER_pt_tot":8,
"DER_sum_pt":9,
"DER_pt_ratio_lep_tau":10,
"DER_met_phi_centrality":11,
"DER_lep_eta_centrality":12,
"PRI_tau_pt":13,
"PRI_tau_eta":14,
"PRI_tau_phi":15,
"PRI_lep_pt":16,
"PRI_lep_eta":17,
"PRI_lep_phi":18,
"PRI_met":19,
"PRI_met_phi":20,
"PRI_met_sumet":21,
"PRI_jet_num":22,
"PRI_jet_leading_pt":23,
"PRI_jet_leading_eta":24,
"PRI_jet_leading_phi":25,
"PRI_jet_subleading_pt":26,
"PRI_jet_subleading_eta":27,
"PRI_jet_subleading_phi":28,
"PRI_jet_all_pt":29}

#divide all samples into four categories based on the PRI_jet_num
feature_0 = features[features[:, feature_dict["PRI_jet_num"]] == 0, :]
feature_1 = features[features[:, feature_dict["PRI_jet_num"]] == 1, :]
feature_2 = features[features[:, feature_dict["PRI_jet_num"]] == 2, :]
feature_3 = features[features[:, feature_dict["PRI_jet_num"]] == 3, :]

#divide all labels into four categories based on the PRI_jet_num
label_0 = labels[features[:, feature_dict["PRI_jet_num"]] == 0]
label_1 = labels[features[:, feature_dict["PRI_jet_num"]] == 1]
label_2 = labels[features[:, feature_dict["PRI_jet_num"]] == 2]
label_3 = labels[features[:, feature_dict["PRI_jet_num"]] == 3]

#choosing features that shoube be dropped by the first category
drop_feature_0 = [feature_dict["DER_deltaeta_jet_jet"],
feature_dict["DER_lep_eta_centrality"],
feature_dict["DER_mass_jet_jet"],
feature_dict["DER_prodeta_jet_jet"],
feature_dict["PRI_jet_all_pt"],
feature_dict["PRI_jet_leading_eta"],
feature_dict["PRI_jet_leading_phi"],
feature_dict["PRI_jet_leading_pt"],
feature_dict["PRI_jet_num"],
feature_dict["PRI_jet_subleading_eta"],
feature_dict["PRI_jet_subleading_phi"],
feature_dict["PRI_jet_subleading_pt"],
feature_dict["PRI_lep_phi"],
feature_dict["PRI_met_phi"],
feature_dict["PRI_tau_phi"],
feature_dict["DER_mass_MMC"],
feature_dict["PRI_met_sumet"]]

#choosing features that shoube be dropped by the second category
drop_feature_1 = [feature_dict["DER_deltaeta_jet_jet"],
feature_dict["DER_lep_eta_centrality"],
feature_dict["DER_mass_jet_jet"],
feature_dict["DER_prodeta_jet_jet"],
feature_dict["PRI_jet_leading_phi"],
feature_dict["PRI_jet_num"],
feature_dict["PRI_jet_subleading_eta"],
feature_dict["PRI_jet_subleading_phi"],
feature_dict["PRI_jet_subleading_pt"],
feature_dict["PRI_lep_phi"],
feature_dict["PRI_met_phi"],
feature_dict["PRI_tau_phi"],
feature_dict["PRI_jet_leading_pt"],
feature_dict["PRI_jet_all_pt"],
feature_dict["DER_mass_MMC"],
feature_dict["PRI_met_sumet"]]

#choosing features that shoube be dropped by the third category
drop_feature_2 = [feature_dict["PRI_jet_leading_phi"],
feature_dict["PRI_jet_num"],
feature_dict["PRI_jet_subleading_phi"],
feature_dict["PRI_lep_phi"],
feature_dict["PRI_met_phi"],
feature_dict["PRI_tau_phi"],
feature_dict["PRI_jet_leading_pt"],
feature_dict["PRI_jet_all_pt"],
feature_dict["DER_mass_MMC"],
feature_dict["PRI_met_sumet"]]

#choosing features that shoube be dropped by the fourth category
drop_feature_3 = [feature_dict["PRI_jet_leading_phi"],
feature_dict["PRI_jet_num"],
feature_dict["PRI_jet_subleading_phi"],
feature_dict["PRI_lep_phi"],
feature_dict["PRI_met_phi"],
feature_dict["PRI_tau_phi"],
feature_dict["PRI_jet_leading_pt"],
feature_dict["PRI_jet_all_pt"],
feature_dict["DER_mass_MMC"],
feature_dict["PRI_met_sumet"]]

#dropping features for all samples of each category
feature_0 = np.delete(feature_0, drop_feature_0, axis = 1)
feature_1 = np.delete(feature_1, drop_feature_1, axis = 1)
feature_2 = np.delete(feature_2, drop_feature_2, axis = 1)
feature_3 = np.delete(feature_3, drop_feature_3, axis = 1)

#applying log-scale or standard normalisation for all samples to accelerate caomputing
feature_0 = log_normal(feature_0)
feature_1 = log_normal(feature_1)
feature_2 = log_normal(feature_2)
feature_3 = log_normal(feature_3)

#build polynomial features 
degree = 3
feature_0_poly = build_poly_feature(feature_0)
feature_1_poly = build_poly_feature(feature_1)
feature_2_poly = build_poly_feature(feature_2)
feature_3_poly = build_poly_feature(feature_3)

#implementing ridge regression and cross validation for each category to find the best weights
k_fold = 8 # the number of divided sets
lamb = 0.01 # regularization coefficient

print('k-fold: ', k_fold, 'polynomial degree: ', degree, 'lambda:', lamb)
print('Start ridge regression with cross validation:')
k_indices_0 = build_k_indices(label_0, k_fold)
w0, right_rate_0 = ridge_regression_cv(label_0, feature_0_poly, lamb, k_indices_0, k_fold)
print('tag 0:', right_rate_0)

k_indices_1 = build_k_indices(label_1, k_fold)
w1, right_rate_1 = ridge_regression_cv(label_1, feature_1_poly, lamb, k_indices_1, k_fold)
print('tag 1:', right_rate_1)

k_indices_2 = build_k_indices(label_2, k_fold)
w2, right_rate_2 = ridge_regression_cv(label_2, feature_2_poly, lamb, k_indices_2, k_fold)
print('tag 2:', right_rate_2)

k_indices_3 = build_k_indices(label_3, k_fold)
w3, right_rate_3 = ridge_regression_cv(label_3, feature_3_poly, lamb, k_indices_3, k_fold)
print('tag 3:', right_rate_3)

###implementing the same processing technique to test data as train data###
label_test, feature_test, id_test = load_csv_data('test.csv') 

test_0 = feature_test[feature_test[:, feature_dict["PRI_jet_num"]] == 0, :]
test_1 = feature_test[feature_test[:, feature_dict["PRI_jet_num"]] == 1, :]
test_2 = feature_test[feature_test[:, feature_dict["PRI_jet_num"]] == 2, :]
test_3 = feature_test[feature_test[:, feature_dict["PRI_jet_num"]] == 3, :]

test_0 = np.delete(test_0, drop_feature_0, axis = 1)
test_1 = np.delete(test_1, drop_feature_1, axis = 1)
test_2 = np.delete(test_2, drop_feature_2, axis = 1)
test_3 = np.delete(test_3, drop_feature_3, axis = 1)

test_0 = log_normal(test_0)
test_1 = log_normal(test_1)
test_2 = log_normal(test_2)
test_3 = log_normal(test_3)

test_0_poly = build_poly_feature(test_0)
test_1_poly = build_poly_feature(test_1)
test_2_poly = build_poly_feature(test_2)
test_3_poly = build_poly_feature(test_3)
###implementing the same processing technique to test data as train data###

###predicting labels and generating submission file###
test_label_0 = predict_labels(w0, test_0_poly)
test_label_1 = predict_labels(w1, test_1_poly)
test_label_2 = predict_labels(w2, test_2_poly)
test_label_3 = predict_labels(w3, test_3_poly)

label_test[np.where(feature_test[:, feature_dict["PRI_jet_num"]] == 0)] = test_label_0
label_test[np.where(feature_test[:, feature_dict["PRI_jet_num"]] == 1)] = test_label_1
label_test[np.where(feature_test[:, feature_dict["PRI_jet_num"]] == 2)] = test_label_2
label_test[np.where(feature_test[:, feature_dict["PRI_jet_num"]] == 3)] = test_label_3

create_csv_submission(id_test, label_test, 'submission.csv')