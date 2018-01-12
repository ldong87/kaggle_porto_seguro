#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from utils import *

def model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
               numl1, numl2,numl3):
    trn_tmp_x = trn_tmp_x.values
    trn_tmp_y = trn_tmp_y.values
    val_tmp_x = val_tmp_x.values
    val_tmp_y = val_tmp_y.values
    tst_x = tst_x.values
    
    trn_tmp_x = MinMaxScaler().fit_transform(trn_tmp_x)
    val_tmp_x = MinMaxScaler().fit_transform(val_tmp_x)
    
    model = SVC(kernel='rbf', degree=6, gamma=0.001, coef0=0.0, tol=0.0001, C=0.01,
          epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=500)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y )
    
    return fit_model.predict_proba(val_tmp_x)[:,1], fit_model.predict_proba(tst_x)[:,1]

trn_tmp_x = trn_tmp_x.values
trn_tmp_y = trn_tmp_y.values
val_tmp_x = val_tmp_x.values
val_tmp_y = val_tmp_y.values
tst_x = tst_x.values

trn_tmp_x = MinMaxScaler().fit_transform(trn_tmp_x)
val_tmp_x = MinMaxScaler().fit_transform(val_tmp_x)
#%%
#model = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, 
#            tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=30, decision_function_shape='ovr',
#            random_state=1987)
model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                  fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=1987, max_iter=1000)

fit_model = model.fit( trn_tmp_x, trn_tmp_y )
pred_trn_tmp = fit_model.predict_proba(val_tmp_x)[:,1]
        
print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
