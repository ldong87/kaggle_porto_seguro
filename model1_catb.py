#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import catboost as catb
from utils import *

def data_categ(x):
    categ = list(x.filter(regex='cat$'))
    categ.extend(list(x.filter(regex='bin$')))
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

def model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
               depth,
              reg_lambda,
              feature_fraction):
    print 'depth = ', depth, ' reg_lambda = ', reg_lambda, 'feature_fraction = ', feature_fraction
    best_iter = 10000
    model = catb.CatBoostClassifier(iterations=best_iter,
                                    learning_rate=0.5,
                                    depth=int(round(depth)),
                                    l2_leaf_reg=reg_lambda,
                                    rsm=feature_fraction,
                                    loss_function='Logloss',
                                    thread_count=16,
                                    random_seed=1987,
                                    use_best_model=True,
                                    od_type='Iter',
                                    od_wait=800,
                                    eval_metric='AUC',
#                                    eval_metric=gini_metric(),
                                    verbose=False)
    cat_f = data_categ(trn_tmp_x)
    fit_model = model.fit( X=trn_tmp_x, y=trn_tmp_y, cat_features=cat_f,
                           use_best_model=True,
                           eval_set=(val_tmp_x,val_tmp_y),
                           verbose=False)
#    print model.get_param('tree_count_')
    
    return fit_model.predict_proba(val_tmp_x)[:,1], fit_model.predict_proba(tst_x)[:,1], best_iter

