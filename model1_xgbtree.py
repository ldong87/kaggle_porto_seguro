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

def model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
               min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight):    
    model = xgb.XGBClassifier(  booster = 'gbtree', 
                                n_estimators=10000,
                                max_depth=int(max_depth),
                                objective="binary:logistic",
                                learning_rate=0.04, 
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
                           eval_metric='auc',
#                           eval_metric=gini_xgb,
                           early_stopping_rounds=1000,
                           verbose=False
                             )
    print model.best_iteration, model.best_score
    return fit_model.predict_proba(val_tmp_x,ntree_limit=model.best_ntree_limit)[:,1], fit_model.predict_proba(tst_x,ntree_limit=model.best_ntree_limit)[:,1], model.best_iteration

def model2_pred(trn_tmp_x,trn_tmp_y,tst_x,
               min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight,best_iter):

    model = xgb.XGBClassifier(  booster = 'gbtree', 
                                n_estimators=best_iter,
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
                           eval_metric='auc',
#                           eval_metric=gini_xgb,
                           verbose=False
                             )

    return fit_model.predict_proba(tst_x)[:,1]

