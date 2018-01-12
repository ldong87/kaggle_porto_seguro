#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
from rgf.sklearn import RGFClassifier
from utils import *

def model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x): 
    best_iter = 1200
    model = RGFClassifier(max_leaf=best_iter, #Try increasing this as a starter
                    algorithm="RGF",
                    loss="Log",
                    l2=0.01,
                    normalize=False,
                    min_samples_leaf=20,
                    learning_rate=0.5,
                    verbose=False)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y )
    
    return fit_model.predict_proba(val_tmp_x)[:,1], fit_model.predict_proba(tst_x)[:,1],best_iter


