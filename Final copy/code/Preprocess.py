#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:22:26 2018

@author: linye
"""

import numpy as np
import pandas as pd

import os

class Preprocess():
    
    def __init__(self, features):
        
        self._features = features
        
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._x_valid = None
        self._y_valid = None
        
        self._y_train_time = None
        self._y_test_time = None
        
    def get_features(self):
        
        return self._features
    
    def _get_samples_index(self, y, sampling):
        
        if type(sampling) == dict:
            sample_sizes = sampling
        elif type(sampling) == str and sampling == 'min':
            sample_sizes = dict([(label, y.value_counts().min()) \
                                 for label in y.unique()])
        elif type(sampling) == str and sampling in list(y.value_counts().index.astype(str)):
            sample_sizes = dict([(label, \
                                  y.value_counts()[int(sampling)])
                                 for label in y.unique()])
        elif type(sampling) == str and 'multi:' in sampling:
            sample_sizes = dict([(label, 
                int(sampling.replace('multi:', '')) * y.value_counts().min())
                    for label in y.unique()])
        elif type(sampling) == int:
            sample_sizes = dict([(label, sampling) for label in y.unique()])
        else:
            print('unknown sampling method or no sampling')
            return y.index

        select_index_all = []
        for label in y.unique():
            label_index = y[y==label].index
            sample_size = sample_sizes[label]
        
            if sample_size <= len(label_index):
                replace = False
            else:
                replace = True
            select_index = np.random.choice(label_index, sample_size, \
                                            replace=replace)
            select_index_all = select_index_all + list(select_index)
    
        return select_index_all
    
    def _get_LSTM_data(self, x_train, y_train, x_test, y_test, sampled_idx, n_steps):
        # training set
        x_train_m = x_train.values
        x_train_list = []
        for idx in sampled_idx:
            int_idx = y_train.index.get_loc(idx)
            x_train_list.append(x_train_m[(int_idx-n_steps+1):int_idx+1])
        
        x_train = np.array(x_train_list)
        y_train = pd.get_dummies(y_train[sampled_idx]).values
        
        # test set
        x_test_m = x_test.values
        x_test_list = []
        for i in range(x_test.shape[0] - n_steps+1):
            x_test_list.append(x_test_m[i: (i + n_steps)])
        x_test = np.array(x_test_list)
        y_test = pd.get_dummies(y_test).values  
        
        return x_train, y_train, x_test, y_test
    
    def filt(self, df_filter):
        
        #check if the filter is of the same index as our dataframes
        if any(self._features.index != df_filter.index):
            raise Exception("The two data files do not contain the same index")
        
        self._features = self._features[df_filter]
        #print(len(self._features))
    
    def split_train_test(self, use_features, train_weight=0.8, n_steps=5, resampling=False):
        
        #self._features = self._features.dropna()
        
        split = int(len(self._features)*train_weight)
        #print(split)
        df_train = self._features.iloc[:split]
        df_test = self._features.iloc[split:]
        
        y_train = df_train.label       
        x_train = df_train[use_features]
        y_test = df_test.label.iloc[n_steps-1:]
        x_test = df_test[use_features]
        
        
        if resampling:
            sampled_idx = self._get_samples_index(y_train.iloc[n_steps-1:], 'min')
        else:
            sampled_idx = list(y_train.iloc[n_steps-1:].index)
            
        self._y_train_time = df_train.loc[sampled_idx].time
        self._y_test_time = df_test.iloc[n_steps-1:].time
        
        self._x_train, self._y_train, self._x_test, self._y_test = self._get_LSTM_data(x_train, y_train, x_test, y_test, sampled_idx, n_steps)
        
    def normalize(self):
        
        x_max = np.max(self._x_train,axis=0)
        x_min = np.min(self._x_train,axis=0)
        self._x_train = (self._x_train - x_min) / (x_max - x_min)
        self._x_test = (self._x_test - x_min) / (x_max - x_min)
    
    def split_valid_test(self, nrow=150):
        
        if nrow >= len(self._x_test):
            raise Exception("Too many valid points")            
        
        self._y_test_time = self._y_test_time[nrow:]
        
        self._x_valid = self._x_test[:nrow]
        self._y_valid = self._y_test[:nrow]
        self._x_test = self._x_test[nrow:]
        self._y_test = self._y_test[nrow:]
    
    def get_train_valid_split(self):
        
        return self._x_train, self._y_train, self._x_test, self._y_test, self._x_valid, self._y_valid
        
    
    def time_to_csv(self, path = os.path.join(os.getcwd(), 'data')):
        
        filename = os.path.join(path, 'y_test_time_intc_up&down_nolesssprd.csv')
        y_test_time_df = pd.DataFrame(self._y_test_time, columns = ['time'])
        y_test_time_df.to_csv(filename, index=False)
        