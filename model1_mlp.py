#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
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
    tst_x = MinMaxScaler().fit_transform(tst_x)
    
    best_iter = 500
    model = MLPClassifier(hidden_layer_sizes=(int(round(numl1)),int(round(numl2)),int(round(numl3)),), activation='relu', solver='lbfgs', alpha=10, batch_size='auto',
               learning_rate='invscaling', learning_rate_init=0.001, power_t=0.5, max_iter=best_iter, shuffle=True,
               random_state=1987, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
               beta_2=0.999, epsilon=1e-08)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y )
    
    return fit_model.predict_proba(val_tmp_x)[:,1], fit_model.predict_proba(tst_x)[:,1], best_iter


