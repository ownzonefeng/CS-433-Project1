import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
import joblib
import sklearn.metrics as metrics
from sklearn.model_selection import GroupShuffleSplit

def rmse(y_true, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return rmse

def lcc(y_true, y_pred):
    lcc, _ = stats.pearsonr(y_true, y_pred)
    return lcc

def srocc(y_true, y_pred):
    srocc, _ = stats.spearmanr(y_true, y_pred)
    return srocc

def CV_Generator(features, labels, group_label, n=8, test_ratio=0.2):
    CV_Group = GroupShuffleSplit(n_splits=n, test_size=test_ratio, random_state=8)
    for train, test in CV_Group.split(features, labels, groups=group_label):
        yield train, test

scorer = {}
scorer['rmse'] = metrics.make_scorer(rmse, greater_is_better=False)
scorer['lcc'] = metrics.make_scorer(lcc, greater_is_better=True)
scorer['srocc'] = metrics.make_scorer(srocc, greater_is_better=True)

features = pd.read_pickle('features.pkl')
labels = pd.read_pickle('labels.pkl')

group_label = np.arange(len(features.index) / 3)
group_label = np.matlib.repmat(group_label,3,1)
group_label = group_label.reshape(-1,1, order='F')

Reg_video = RandomForestRegressor(random_state=8, n_jobs=-1)

parameters_grid_GCV_3MET = {}

parameters_grid_GCV_3MET['n_estimators'] = [163, 162, 164]
parameters_grid_GCV_3MET['criterion'] = ['mse']
parameters_grid_GCV_3MET['max_depth'] = [10,11,9]
parameters_grid_GCV_3MET['max_depth'].append(None)
parameters_grid_GCV_3MET['min_samples_split'] = [2]
parameters_grid_GCV_3MET['min_samples_leaf'] = [2]
parameters_grid_GCV_3MET['max_features'] = ['sqrt', 'auto']
parameters_grid_GCV_3MET['bootstrap'] = [True]
parameters_grid_GCV_3MET['verbose'] = [0]
parameters_grid_GCV_3MET['oob_score'] = [True]

parameters_grid_search_GCV_3MET = GridSearchCV(estimator = Reg_video, param_grid = parameters_grid_GCV_3MET, 
                          cv = CV_Generator(features, labels, group_label), n_jobs = -1, verbose = 1, return_train_score=True, 
                                      error_score = np.nan, scoring = scorer, refit = 'lcc', iid=False)
parameters_grid_search_GCV_3MET.fit(features, labels)
#joblib.dump(parameters_grid_search_GCV_3MET, 'parameters_grid_search_GCV_3MET.sav')

