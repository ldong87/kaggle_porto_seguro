#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
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
    
    model = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                                  max_iter=500, tol=1e-4, shuffle=True, verbose=0, epsilon=0.1, n_jobs=16, 
                                  random_state=1987, learning_rate='invscaling', eta0=0.01, power_t=0.5, class_weight=None, 
                                  warm_start=False, average=False, n_iter=None)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y )
    
    return fit_model.predict_proba(val_tmp_x)[:,1], fit_model.predict_proba(tst_x)[:,1]

#%%
model = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.1, l1_ratio=0.55, fit_intercept=True,
                                  max_iter=1500, tol=1e-4, shuffle=True, verbose=0, epsilon=0.1, n_jobs=16, 
                                  random_state=1987, learning_rate='invscaling', eta0=0.001, power_t=0.5, class_weight=None, 
                                  warm_start=False, average=False, n_iter=None)

fit_model = model.fit( trn_tmp_x, trn_tmp_y )
pred_trn_tmp = fit_model.predict_proba(val_tmp_x)[:,1]

print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
