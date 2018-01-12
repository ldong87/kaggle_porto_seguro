#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from utils import *

def model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
               C):
    trn_tmp_x = trn_tmp_x.values
    trn_tmp_y = trn_tmp_y.values
    val_tmp_x = val_tmp_x.values
    val_tmp_y = val_tmp_y.values
    tst_x = tst_x.values
    
    trn_tmp_x = MinMaxScaler().fit_transform(trn_tmp_x)
    val_tmp_x = MinMaxScaler().fit_transform(val_tmp_x)
    tst_x = MinMaxScaler().fit_transform(tst_x)
    
    best_iter = 800
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=C, 
                               fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 
                               solver='liblinear', max_iter=best_iter, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y )
    
    return fit_model.predict_proba(val_tmp_x)[:,1], fit_model.predict_proba(tst_x)[:,1], best_iter

