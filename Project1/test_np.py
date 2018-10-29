#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 07:45:12 2018

@author: wentao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from Myhelper import *
from proj1_helpers import *
from implementations import *

labels, features, ids = load_csv_data('train.csv') 
labels = (labels + 1) /2 

feature_dict = {"DER_mass_MMC":0,
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

feature_0 = features[features[:, feature_dict["PRI_jet_num"]] == 0, :]


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

feature_0 = np.delete(feature_0, drop_feature_0, axis = 1)

feature_0 = log_normal(feature_0)

feature_0_poly = build_poly_feature(feature_0)