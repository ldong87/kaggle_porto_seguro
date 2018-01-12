#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import *
    
def gini_metric(preds, labels):
    gini_score = -eval_gini(labels, preds)
    return 'gini', gini_score, False

def model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
               max_depth,
              min_child_samples,
              min_child_weight,
              colsample_bytree,
              subsample,
              subsample_freq,
              num_leaves,
              alpha,
              scale_pos_weight,
              drop_rate,
              skip_drop):
    
    model = lgb.LGBMClassifier( boosting_type = 'dart', 
                                n_estimators=500,
                                max_depth=int(round(max_depth)),
                                objective="binary_logloss",
                                learning_rate=0.01, 
                                num_leaves=int(num_leaves),
                                subsample=subsample, # bagging_fraction
                                subsample_freq=int(subsample_freq),
                                min_child_samples=int(min_child_samples),
                                min_child_weight=min_child_weight, #min_sum_hessian_in_leaf
                                colsample_bytree=colsample_bytree, # feature fraction
                                scale_pos_weight=scale_pos_weight,
                                reg_alpha=alpha,
                                reg_lambda=1.3,
                                drop_rate=drop_rate,
                                skip_drop=skip_drop,
                                max_drop=50,
                                uniform_drop=False,
                                n_jobs=8)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y, 
                           eval_set=[(val_tmp_x,val_tmp_y)],
                           eval_metric=gini_metric,
                           early_stopping_rounds=50,
                           verbose=False)
    
    return fit_model.predict_proba(val_tmp_x,num_iteration=model.best_iteration_)[:,1], fit_model.predict_proba(tst_x,num_iteration=model.best_iteration_)[:,1], model.best_iteration_
