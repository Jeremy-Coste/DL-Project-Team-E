#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: meihuaren
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#-- import data
orderbook_cols = ['{}_{}_{}'.format(s,t,l) for l in range(1,6) for s in ['ask','bid'] for t in ['price', 'vol'] ]
orderbook_ori = pd.read_csv('INTC_2012-06-21_34200000_57600000_orderbook_5.csv', \
                            header = None, names = orderbook_cols)
orderbook1 = orderbook_ori.copy()
orderbook1['mid_price'] = (orderbook1.iloc[:,0] + orderbook1.iloc[:,2]) / 2
orderbook1['mid_price_mov'] = np.sign(orderbook1['mid_price'].shift(-1)-orderbook1['mid_price'])
orderbook2 = orderbook1.dropna()

scaler = StandardScaler()
x_all_array = scaler.fit_transform(orderbook2.iloc[:,:len(orderbook_cols)])
orderbook = orderbook2.copy()
orderbook.iloc[:,:len(orderbook_cols)] = x_all_array

train_weight = 0.8
cv_weight = 0.1
split1 = int(orderbook.shape[0] * train_weight)
split2 = int(orderbook.shape[0] * cv_weight)
df_train = orderbook[:split1]
df_cv = orderbook[split1:split1+split2]
df_test = orderbook[split1+split2:]
x_train = df_train.iloc[:,:len(orderbook_cols)]
y_train = df_train.iloc[:,-1]
x_cv = df_cv.iloc[:,:len(orderbook_cols)]
y_cv = df_cv.iloc[:,-1]
x_test = df_test.iloc[:,:len(orderbook_cols)]
y_test = df_test.iloc[:,-1]
x_all = orderbook.iloc[:,:len(orderbook_cols)]
#y_all = orderbook.iloc[:,-1]

#-- feature selection
method = 'select_from_model_extratrees'

'''
see more at:
http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
use train data for feature selection in order to avoid look ahead bias
'''
#-- method 2-1 select k best: f_classif
if method == 'select_k_best_f_classif':
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    model_fs = SelectKBest(f_classif, k=15).fit(x_train, y_train) # grid search for the parameter

#-- method 2-2 SelectFdr: f_classif # no use
if method == 'select_fdr_f_classif':
    from sklearn.feature_selection import SelectFdr
    from sklearn.feature_selection import f_classif
    model_fs = SelectFdr(f_classif, alpha=1e-7).fit(x_train, y_train) # grid search for the parameter
    
#-- method 2-3 SelectFwe: f_classif # no use==
if method == 'select_fwe_f_classif':
    from sklearn.feature_selection import SelectFwe
    from sklearn.feature_selection import f_classif
    model_fs = SelectFwe(f_classif, alpha=0.0001).fit(x_train, y_train)

#-- method 3 RFECV: SVC # too slow
if method == 'rfecv_svc':
    from sklearn.feature_selection import RFECV
    from sklearn.svm import SVC
    svc = SVC(kernel="linear")
    model_fs_pre = RFECV(estimator=svc, step=1, cv=5)
    model_fs = model_fs_pre.fit(x_train, y_train)

#-- method 4-1 select from model: LinearSVC (L1-based) # too slow?
if method == 'select_from_model_linear_svc':
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    model_fs_pre = LinearSVC(C=0.01, penalty="l1", dual=False) # grid search for the parameter
    model_fs_pre = model_fs_pre.fit(x_train, y_train)
    model_fs = SelectFromModel(model_fs_pre, prefit=True)

#-- method 4-2 select from model: ExtraTrees # fast and select 3 features for n=50
if method == 'select_from_model_extratrees': 
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    model_fs_pre = ExtraTreesClassifier(n_estimators=50) # grid search for the parameter
    model_fs_pre = model_fs_pre.fit(x_train, y_train)
    model_fs = SelectFromModel(model_fs_pre, prefit=True)


x_all_new = model_fs.transform(x_all)
x_all_new_icol = list(model_fs.get_support(indices=True))
x_all_new_df = orderbook.iloc[:,x_all_new_icol]
orderbook_new = pd.concat([x_all_new_df,orderbook.iloc[:,-1]],axis = 1)

new_features_resultpath = '/Users/meihuaren/personal/OR_2018fall/Courses/E4720 Deep Learning/project_coding/Team E_code/'
filename = new_features_resultpath + 'ob_new_' + method + '.csv'
orderbook_new.to_csv(filename)