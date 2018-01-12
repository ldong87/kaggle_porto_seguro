#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from utils import *

def model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x):
    trn_tmp_x = trn_tmp_x.values
    trn_tmp_y = trn_tmp_y.values
    val_tmp_x = val_tmp_x.values
    val_tmp_y = val_tmp_y.values
    tst_x = tst_x.values
    
    trn_tmp_x = MinMaxScaler().fit_transform(trn_tmp_x)
    val_tmp_x = MinMaxScaler().fit_transform(val_tmp_x)
    
    model = KNeighborsClassifier( n_neighbors=int(np.sqrt(trn_tmp_x.shape[0])), weights='distance', n_jobs=8 )
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y )
    
    return fit_model.predict_proba(val_tmp_x)[:,1], fit_model.predict_proba(tst_x)[:,1]

