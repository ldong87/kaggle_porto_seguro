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


#%% xgbtree
import model1_xgbtree as m1xgbtree

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

    pred_trn_tmp, pred_tst_tmp, best_iter = m1xgbtree.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x_meta,
                                                      min_child_weight_,colsample_bytree_,max_depth_,subsample_,gamma_,
                                                      alpha_,scale_pos_weight_)
    print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
    pred_trn[cv_ind[flag_valid[:,j]]] = pred_trn_tmp
    
    return [pred_trn, pred_tst_tmp, best_iter]

def one_pass(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight):
    global pred_trn, pred_tst, best_iter
    global min_child_weight_,colsample_bytree_,max_depth_,subsample_,gamma_,alpha_,scale_pos_weight_
    min_child_weight_ = min_child_weight
    colsample_bytree_ = colsample_bytree
    max_depth_ = max_depth
    subsample_ = subsample
    gamma_ = gamma
    alpha_ = alpha
    scale_pos_weight_ = scale_pos_weight
    
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

#modelBO = BO(one_pass, {    'min_child_weight': (1, 20),
#                            'colsample_bytree': (0.1, 1),
#                            'max_depth': (1, 10),
#                            'subsample': (0.5, 1),
#                            'gamma': (0, 20),
#                            'alpha': (0, 20),
#                            'scale_pos_weight': (1,4)
#                            }, random_state=201)
#modelBO.maximize(init_points=50, n_iter=50, acq='rnd')
#print modelBO.res['max']['max_params']['min_child_weight'],\
#      modelBO.res['max']['max_params']['colsample_bytree'],\
#      modelBO.res['max']['max_params']['max_depth'],\
#      modelBO.res['max']['max_params']['subsample'],\
#      modelBO.res['max']['max_params']['gamma'],\
#      modelBO.res['max']['max_params']['alpha'],\
#      modelBO.res['max']['max_params']['scale_pos_weight']
#one_pass(modelBO.res['max']['max_params']['min_child_weight'],
#         modelBO.res['max']['max_params']['colsample_bytree'],
#         modelBO.res['max']['max_params']['max_depth'],
#         modelBO.res['max']['max_params']['subsample'],
#         modelBO.res['max']['max_params']['gamma'],
#         modelBO.res['max']['max_params']['alpha'],
#         modelBO.res['max']['max_params']['scale_pos_weight'])

one_pass(10.3752371952, 0.987316075503, 4.57475064593, 0.537517379712, 6.85815456978, 12.7923301534, 1.53280643886)
#min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha, scale_pos_weight

print "\nDouble check Gini for full training set ", ':', eval_gini(trn_y, pred_trn)

#pred_tst_tmp = m1xgbtree.model2_pred(trn_x,trn_y,tst_x,
#                                                   modelBO.res['max']['max_params']['min_child_weight'],
#                                                 modelBO.res['max']['max_params']['colsample_bytree'],
#                                                 modelBO.res['max']['max_params']['max_depth'],
#                                                 modelBO.res['max']['max_params']['subsample'],
#                                                 modelBO.res['max']['max_params']['gamma'],
#                                                 modelBO.res['max']['max_params']['alpha'],
#                                                 modelBO.res['max']['max_params']['scale_pos_weight'],
#                                                 best_iter)

t1 = timer()
print 'Timer xgbtree:', t1-t0


save_pred(train.id,test.id,1,pred_trn,pred_tst,'m2_xgbtree')
save_sub(test.id,pred_tst,'xgbtree2')
