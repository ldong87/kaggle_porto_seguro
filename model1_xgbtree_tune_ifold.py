#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from utils import *
    
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def model_run(min_child_weight,
              colsample_bytree,
              max_depth,
              subsample,
              gamma,
              alpha,
              scale_pos_weight):
#    model = xgb.XGBClassifier(  booster = 'gbtree', 
#                                n_estimators=2000,
#                                max_depth=4,
#                                objective="binary:logistic",
#                                learning_rate=0.07, 
#                                subsample=.8,
#                                min_child_weight=6,
#                                colsample_bytree=.8,
#                                scale_pos_weight=1.6,
#                                gamma=10,
#                                reg_alpha=8,
#                                reg_lambda=1.3,
#                                n_jobs=32
#                         )
    model = xgb.XGBClassifier(  booster = 'gbtree', 
                                n_estimators=2000,
                                max_depth=int(max_depth),
                                objective="binary:logistic",
                                learning_rate=0.05, 
                                subsample=subsample,
                                min_child_weight=int(min_child_weight),
                                colsample_bytree=colsample_bytree,
                                scale_pos_weight=scale_pos_weight,
                                gamma=gamma,
                                reg_alpha=alpha,
                                reg_lambda=1.3,
                                n_jobs=8
                         )
    fit_model = model.fit( trn_tmp_x, trn_tmp_y, 
                           eval_set=[(val_tmp_x,val_tmp_y)],
                           eval_metric=gini_xgb,
                           early_stopping_rounds=100,
                           verbose=False
                             )
#    global pred
    pred.append(fit_model.predict_proba(val_tmp_x)[:,1])
    
    return eval_gini(val_tmp_y, pred[-1])

def model_pred(trn_x,trn_y,val_x,val_y,tst_x,
               min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight):
#    global trn_tmp_x
#    global trn_tmp_y
#    global val_tmp_x
#    global val_tmp_y
    trn_tmp_x = trn_x
    trn_tmp_y = trn_y
    val_tmp_x = val_x
    val_tmp_y = val_y
    
#    global pred
#    pred = []
    
    model = xgb.XGBClassifier(  booster = 'gbtree', 
                                n_estimators=2000,
                                max_depth=int(max_depth),
                                objective="binary:logistic",
                                learning_rate=0.05, 
                                subsample=subsample,
                                min_child_weight=int(min_child_weight),
                                colsample_bytree=colsample_bytree,
                                scale_pos_weight=scale_pos_weight,
                                gamma=gamma,
                                reg_alpha=alpha,
                                reg_lambda=1.3,
                                n_jobs=8
                         )
        
#    model = xgb.XGBClassifier(  booster = 'gbtree', 
#                                n_estimators=2000,
#                                max_depth=int(modelBO.res['max']['max_params']['max_depth']),
#                                objective="binary:logistic",
#                                learning_rate=0.05, 
#                                subsample=modelBO.res['max']['max_params']['subsample'],
#                                min_child_weight=int(modelBO.res['max']['max_params']['min_child_weight']),
#                                colsample_bytree=modelBO.res['max']['max_params']['colsample_bytree'],
#                                scale_pos_weight=modelBO.res['max']['max_params']['scale_pos_weight'],
#                                gamma=modelBO.res['max']['max_params']['gamma'],
#                                reg_alpha=modelBO.res['max']['max_params']['alpha'],
#                                reg_lambda=1,
#                                n_jobs=8
#                         )
    fit_model = model.fit( trn_tmp_x, trn_tmp_y, 
                           eval_set=[(val_tmp_x,val_tmp_y)],
                           eval_metric=gini_xgb,
                           early_stopping_rounds=100,
                           verbose=False
                             )
#    max_pred_ind = (np.in1d(modelBO.res['all']['values'],modelBO.res['max']['max_val'])).argmax()
#    return pred[max_pred_ind], fit_model.predict_proba(tst_x,ntree_limit=model.best_ntree_limit)[:,1]
    return fit_model.predict_proba(val_tmp_x,ntree_limit=model.best_ntree_limit)[:,1], fit_model.predict_proba(tst_x,ntree_limit=model.best_ntree_limit)[:,1]

