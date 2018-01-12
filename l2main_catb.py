#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:54:49 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import cPickle as pk
from utils import *
import multiprocessing
from timeit import default_timer as timer
from bayes_opt_test import BayesianOptimization as BO

np.random.seed(0)

#with open('mis.pkl','rb') as f:
#    nrow_trn,nrow_tst = pk.load(f)

nfold = 5 # must be greater than 2
with open('data.pkl','rb') as f:
        [train, test] = pk.load(f)
with open('meta_combo.pkl','rb') as f:
    meta_trn_nfold, meta_tst_nfold = pk.load(f)

trn_x = meta_trn_nfold
tst_x = meta_tst_nfold

#trn_x0 = train.loc[:,'ps_ind_01':]
#tst_x0 = test.loc[:,'ps_ind_01':]

trn_x0 = pick_feat(train.loc[:,'ps_ind_01':])
tst_x0 = pick_feat(test.loc[:,'ps_ind_01':])

trn_y = train.target


nrow_trn = train.shape[0]
nrow_tst = test.shape[0]
pred_trn = np.zeros(nrow_trn)
pred_tst = np.zeros(nrow_tst)
best_iter = 0

flag_valid, cv_ind = create_valid(nfold, nrow_trn)


#%% catb
import model1_catb as m1catb

def forloop(j):
    print 'jfold: ', j
    
    trn_x_meta = pd.concat([trn_x0,trn_x[j]],axis=1)
    tst_x_meta = pd.concat([tst_x0,tst_x[j]],axis=1)
    
#    trn_x_meta = trn_x[j]
#    tst_x_meta = tst_x[j]
    
    trn_tmp_x = trn_x_meta.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x_meta.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]

    pred_trn_tmp, pred_tst_tmp, best_iter = m1catb.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x_meta,
                                                      depth_,reg_lambda_,feature_fraction_)
    print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
    pred_trn[cv_ind[flag_valid[:,j]]] = pred_trn_tmp
    
    return [pred_trn, pred_tst_tmp, best_iter]

def one_pass(depth,reg_lambda,feature_fraction):
    global pred_trn, pred_tst, best_iter
    global depth_,reg_lambda_,feature_fraction_
    depth_ = depth
    reg_lambda_ = reg_lambda
    feature_fraction_ = feature_fraction
    
    n_core = int(nfold*10)
    pool = multiprocessing.Pool(n_core)
    result_combo = pool.map(forloop, range(nfold))   
    pool.close()
    pool.join()
    pred_trn_separate = np.transpose(np.array([result_combo[i][0] for i in xrange(nfold)]))
    pred_tst = np.mean(np.transpose(np.array([result_combo[i][1] for i in xrange(nfold)])), axis=1)
    best_iter = int(np.mean(np.array([result_combo[i][2] for i in xrange(nfold)])))
    
    for i in xrange(nfold):
        pred_trn[cv_ind[flag_valid[:,i]]] = pred_trn_separate[cv_ind[flag_valid[:,i]],i]

    return eval_gini(trn_y, pred_trn)

t0 = timer()

#modelBO = BO(one_pass, {    'depth': (2,10),
#                            'reg_lambda': (0,20),
#                            'feature_fraction': (0.5,1)
#                            }, random_state=1987)
#modelBO.maximize(init_points=50, n_iter=50, acq='rnd')
#print modelBO.res['max']['max_params']['depth'],\
#         modelBO.res['max']['max_params']['reg_lambda'],\
#         modelBO.res['max']['max_params']['feature_fraction']
#one_pass(modelBO.res['max']['max_params']['depth'],
#         modelBO.res['max']['max_params']['reg_lambda'],
#         modelBO.res['max']['max_params']['feature_fraction'])

one_pass(2, 10.9735, 0.8024)
print "\nDouble check Gini for full training set ", ':', eval_gini(trn_y, pred_trn)

t1 = timer()
print 'Timer catb:', t1-t0


save_pred(train.id,test.id,1,pred_trn,pred_tst,'m2_catb')
save_sub(test.id,pred_tst,'catb2')
