#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import catboost as catb
from bayes_opt_test import BayesianOptimization as BO
from utils import *

def data_categ(x):
    categ = list(x.filter(regex='cat'))
    categ.extend(list(x.filter(regex='bin')))
    cat_feature = np.squeeze(np.array(np.where(np.in1d(list(x),categ))))
    return cat_feature
    
class gini_metric(object):
    def get_final_error(self,error,weight):
        return error / (weight + 1e-38)
    def is_max_optimal(self):
        return True
    def evaluate(self,approxes,target,weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = eval_gini(target,approx)
        weight_sum = float(len(approx))
        return error_sum, weight_sum

def model_run(depth,
              reg_lambda,
              feature_fraction):
    
    model = catb.CatBoostClassifier(iterations=2000,
                                    learning_rate=0.05,
                                    depth=int(round(depth)),
                                    l2_leaf_reg=reg_lambda,
                                    rsm=feature_fraction,
                                    loss_function='Logloss',
                                    thread_count=16,
                                    random_seed=1987,
                                    use_best_model=True,
                                    od_type='Iter',
                                    od_wait=100,
                                    eval_metric=gini_metric(),
                                    verbose=False)
    cat_f = data_categ(trn_tmp_x)
    fit_model = model.fit( X=trn_tmp_x, y=trn_tmp_y, cat_features=cat_f,
                           use_best_model=True,
                           eval_set=(val_tmp_x,val_tmp_y),
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
    
    modelBO = BO(model_run, {   'depth': (3,10),
                                'reg_lambda': (0,20),
                                'feature_fraction': (0.5,1)
                                }, random_state=1987)
#    modelBO.explore({'depth': [6],
#                     'reg_lambda': [14],
#                     'feature_fraction': [1]})
    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
    
    model = catb.CatBoostClassifier(iterations=2000,
                                    learning_rate=0.05,
                                    depth=int(round(modelBO.res['max']['max_params']['depth'])),
                                    l2_leaf_reg=modelBO.res['max']['max_params']['reg_lambda'],
                                    rsm=modelBO.res['max']['max_params']['feature_fraction'],
                                    loss_function='Logloss',
                                    thread_count=16,
                                    random_seed=1987,
                                    use_best_model=True,
                                    od_type='Iter',
                                    od_wait=100,
                                    eval_metric=gini_metric(),
                                    verbose=False)
    cat_f = data_categ(trn_tmp_x)
    fit_model = model.fit( X=trn_tmp_x, y=trn_tmp_y, cat_features=cat_f,
                           use_best_model=True,
                           eval_set=(val_tmp_x,val_tmp_y),
                           verbose=False)
    
    max_pred_ind = (np.in1d(modelBO.res['all']['values'],modelBO.res['max']['max_val'])).argmax()
    return pred[max_pred_ind], fit_model.predict_proba(tst_x)[:,1]

