#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from bayes_opt_test import BayesianOptimization as BO
from utils import *
    
def gini_metric(preds, labels):
    gini_score = -eval_gini(labels, preds)
    return 'gini', gini_score, False

def model_run(max_depth,
              min_child_samples,
              min_child_weight,
              colsample_bytree,
              subsample,
              subsample_freq,
              num_leaves,
              alpha,
              scale_pos_weight):
    
    model = lgb.LGBMClassifier( boosting_type = 'gbdt', 
                                n_estimators=2000,
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
                                n_jobs=8)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y, 
                           eval_set=[(val_tmp_x,val_tmp_y)],
                           eval_metric=gini_metric,
                           early_stopping_rounds=100,
                           verbose=False)
#    global pred
    pred.append(fit_model.predict_proba(val_tmp_x)[:,1])
    
    return eval_gini(val_tmp_y, pred[-1])

def model_pred(trn_x,trn_y,val_x,val_y,tst_x):
    global trn_tmp_x
    global trn_tmp_y
    global val_tmp_x
    global val_tmp_y
    trn_tmp_x = trn_x
    trn_tmp_y = trn_y
    val_tmp_x = val_x
    val_tmp_y = val_y
    
    global pred
    pred = []
    
    modelBO = BO(model_run, {   'max_depth': (1,10),
                                'min_child_samples': (10,600),
                                'min_child_weight': (0.001, 20),
                                'colsample_bytree': (0.1, 1),
                                'subsample': (0.5, 1),
                                'subsample_freq': (10,100),
                                'num_leaves': (10,600),
                                'alpha': (0, 20),
                                'scale_pos_weight': (1,4)
                                }, random_state=1987)
    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
    
    model = lgb.LGBMClassifier( boosting_type = 'gbdt', 
                                n_estimators=2000,
                                max_depth=int(round(modelBO.res['max']['max_params']['max_depth'])),
                                objective="binary_logloss",
                                learning_rate=0.01, 
                                num_leaves=int(modelBO.res['max']['max_params']['num_leaves']),
                                subsample=modelBO.res['max']['max_params']['subsample'], # bagging_fraction
                                subsample_freq=int(modelBO.res['max']['max_params']['subsample_freq']),
                                min_child_samples=int(modelBO.res['max']['max_params']['min_child_samples']),
                                min_child_weight=modelBO.res['max']['max_params']['min_child_weight'], #min_sum_hessian_in_leaf
                                colsample_bytree=modelBO.res['max']['max_params']['colsample_bytree'], # feature fraction
                                scale_pos_weight=modelBO.res['max']['max_params']['scale_pos_weight'],
                                reg_alpha=modelBO.res['max']['max_params']['alpha'],
                                reg_lambda=1.3,
                                n_jobs=8)
    
    fit_model = model.fit( trn_tmp_x, trn_tmp_y, 
                           eval_set=[(val_tmp_x,val_tmp_y)],
                           eval_metric=gini_metric,
                           early_stopping_rounds=100,
                           verbose=False)
    
    max_pred_ind = (np.in1d(modelBO.res['all']['values'],modelBO.res['max']['max_val'])).argmax()
    return pred[max_pred_ind], fit_model.predict_proba(tst_x,num_iteration=model.best_iteration_)[:,1]

