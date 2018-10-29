import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv('train.csv', index_col = 0)
train['Prediction'] = train['Prediction'].replace({'s': 1, 'b': -1})
feature_0 = train[train['PRI_jet_num'] == 0]
feature_1 = train[train['PRI_jet_num'] == 1]
feature_2 = train[train['PRI_jet_num'] == 2]
feature_3 = train[train['PRI_jet_num'] == 3]

drop_feature_0 = ['DER_deltaeta_jet_jet',
                  'DER_lep_eta_centrality',
                  'DER_mass_jet_jet',
                  'DER_prodeta_jet_jet',
                  'PRI_jet_all_pt',
                  'PRI_jet_leading_eta',
                  'PRI_jet_leading_phi',
                  'PRI_jet_leading_pt',
                  'PRI_jet_num',
                  'PRI_jet_subleading_eta',
                  'PRI_jet_subleading_phi',
                  'PRI_jet_subleading_pt',
                  'PRI_lep_phi',
                  'PRI_met_phi',
                  'PRI_tau_phi',
                  'DER_mass_MMC', #good corr with DER_mass_vis
                  'PRI_met_sumet',
                 ]

drop_feature_1 = ['DER_deltaeta_jet_jet',
                 'DER_lep_eta_centrality',
                  'DER_mass_jet_jet',
                  'DER_prodeta_jet_jet',
                  'PRI_jet_leading_phi',
                  'PRI_jet_num',
                  'PRI_jet_subleading_eta',
                  'PRI_jet_subleading_phi',
                  'PRI_jet_subleading_pt',
                  'PRI_lep_phi',
                  'PRI_met_phi',
                  'PRI_tau_phi',
                  'PRI_jet_leading_pt',
                  'PRI_jet_all_pt',
                  'DER_mass_MMC',
                  'PRI_met_sumet'
                 ]
drop_feature_2 = ['PRI_jet_leading_phi',
                  'PRI_jet_num',
                  'PRI_jet_subleading_phi',
                  'PRI_lep_phi',
                  'PRI_met_phi',
                  'PRI_tau_phi',
                  'PRI_jet_leading_pt',
                  'PRI_jet_all_pt',
                  'DER_mass_MMC',
                  'PRI_met_sumet'
                 ]

drop_feature_3 = ['PRI_jet_leading_phi',
                  'PRI_jet_num',
                  'PRI_jet_subleading_phi',
                  'PRI_lep_phi',
                  'PRI_met_phi',
                  'PRI_tau_phi',
                 'PRI_jet_leading_pt',
                  'PRI_jet_all_pt',
                  'DER_mass_MMC',
                  'PRI_met_sumet']

feature_0 = feature_0.drop(columns = drop_feature_0)
feature_1 = feature_1.drop(columns = drop_feature_1)
feature_2 = feature_2.drop(columns = drop_feature_2)
feature_3 = feature_3.drop(columns = drop_feature_3)

for column in feature_0.columns[1:]:
    if np.all(feature_0[column].values >= 0):
        print(column, '-log')
        feature_0.loc[:,column] = np.log(feature_0[column] + 0.0001)
    else:
        print(column, '-normal')
        feature_0.loc[:,column] = (feature_0[column] - np.mean(feature_0[column].values)) / np.std(feature_0[column].values)
  
for column in feature_1.columns[1:]:
    if np.all(feature_1[column].values >= 0):
        print(column, '-log')
        feature_1.loc[:,column] = np.log(feature_1[column] + 0.0001)
    else:
        print(column, '-normal')
        feature_1.loc[:,column] = (feature_1[column] - np.mean(feature_1[column].values)) / np.std(feature_1[column].values)

for column in feature_2.columns[1:]:
    if np.all(feature_2[column].values >= 0):
        print(column, '-log')
        feature_2.loc[:,column] = np.log(feature_2[column] + 0.0001)
    else:
        print(column, '-normal')
        feature_2.loc[:,column] = (feature_2[column] - np.mean(feature_2[column].values)) / np.std(feature_2[column].values)
                      
for column in feature_3.columns[1:]:
    if np.all(feature_3[column].values >= 0):
        print(column, '-log')
        feature_3.loc[:,column] = np.log(feature_3[column] + 0.0001)
    else:
        print(column, '-normal')
        feature_3.loc[:,column] = (feature_3[column] - np.mean(feature_3[column].values)) / np.std(feature_3[column].values)
        
poly = PolynomialFeatures(2)
x_0 = poly.fit_transform(feature_0.drop(columns='Prediction'))
x_1 = poly.fit_transform(feature_1.drop(columns='Prediction'))
x_2 = poly.fit_transform(feature_2.drop(columns='Prediction'))
x_3 = poly.fit_transform(feature_3.drop(columns='Prediction'))

clf_0 = LogisticRegression(max_iter = 50000, n_jobs = -1, verbose=1, fit_intercept=False, solver = 'lbfgs').fit(x_0, feature_0['Prediction'])
clf_1 = LogisticRegression(max_iter = 50000, n_jobs = -1, verbose=1, fit_intercept=False, solver = 'lbfgs').fit(x_1, feature_1['Prediction'])
clf_2 = LogisticRegression(max_iter = 50000, n_jobs = -1, verbose=1, fit_intercept=False, solver = 'lbfgs').fit(x_2, feature_2['Prediction'])
clf_3 = LogisticRegression(max_iter = 50000, n_jobs = -1, verbose=1, fit_intercept=False, solver = 'lbfgs').fit(x_3, feature_3['Prediction'])

test = pd.read_csv('test.csv', index_col= 0)

test_0 = test[test['PRI_jet_num'] == 0]
test_1 = test[test['PRI_jet_num'] == 1]
test_2 = test[test['PRI_jet_num'] == 2]
test_3 = test[test['PRI_jet_num'] == 3]

test_0 = test_0.drop(columns = drop_feature_0)
test_1 = test_1.drop(columns = drop_feature_1)
test_2 = test_2.drop(columns = drop_feature_2)
test_3 = test_3.drop(columns = drop_feature_3)

for column in test_0.columns[1:]:
    if np.all(test_0[column].values >= 0):
        print(column, '-log')
        test_0.loc[:,column] = np.log(test_0[column] + 0.0001)
    else:
        print(column, '-normal')
        test_0.loc[:,column] = (test_0[column] - np.mean(test_0[column].values)) / np.std(test_0[column].values)

for column in test_1.columns[1:]:
    if np.all(test_1[column].values >= 0):
        print(column, '-log')
        test_1.loc[:,column] = np.log(test_1[column] + 0.0001)
    else:
        print(column, '-normal')
        test_1.loc[:,column] = (test_1[column] - np.mean(test_1[column].values)) / np.std(test_1[column].values)

for column in test_2.columns[1:]:
    if np.all(test_2[column].values >= 0):
        print(column, '-log')
        test_2.loc[:,column] = np.log(test_2[column] + 0.0001)
    else:
        print(column, '-normal')
        test_2.loc[:,column] = (test_2[column] - np.mean(test_2[column].values)) / np.std(test_2[column].values)

for column in test_3.columns[1:]:
    if np.all(test_3[column].values >= 0):
        print(column, '-log')
        test_3.loc[:,column] = np.log(test_3[column] + 0.0001)
    else:
        print(column, '-normal')
        test_3.loc[:,column] = (test_3[column] - np.mean(test_3[column].values)) / np.std(test_3[column].values)

predict_label_0 = clf_0.predict(poly.fit_transform(test_0.drop(columns='Prediction')))
predict_label_1 = clf_1.predict(poly.fit_transform(test_1.drop(columns='Prediction')))
predict_label_2 = clf_2.predict(poly.fit_transform(test_2.drop(columns='Prediction')))
predict_label_3 = clf_3.predict(poly.fit_transform(test_3.drop(columns='Prediction')))

sub = pd.read_csv('sample-submission.csv', index_col=0)

sub.loc[test_0.index,'Prediction'] = predict_label_0
sub.loc[test_1.index,'Prediction'] = predict_label_1
sub.loc[test_2.index,'Prediction'] = predict_label_2
sub.loc[test_3.index,'Prediction'] = predict_label_3

pd.DataFrame.to_csv(sub, 'sub_server.csv')                           
