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
import model1_lgbdt as m1lgbdt

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

    pred_trn_tmp, pred_tst_tmp, best_iter = m1lgbdt.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x_meta,
                                                          max_depth_,
                                                          min_child_samples_,
                                                          min_child_weight_,
                                                          colsample_bytree_,
                                                          subsample_,
                                                          subsample_freq_,
                                                          num_leaves_,
                                                          alpha_,
                                                          scale_pos_weight_)
    print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
    pred_trn[cv_ind[flag_valid[:,j]]] = pred_trn_tmp
    
    return [pred_trn, pred_tst_tmp, best_iter]

def one_pass( max_depth,
              min_child_samples,
              min_child_weight,
              colsample_bytree,
              subsample,
              subsample_freq,
              num_leaves,
              alpha,
              scale_pos_weight):
    global pred_trn, pred_tst, best_iter
    global max_depth_, min_child_samples_, min_child_weight_, colsample_bytree_, subsample_, subsample_freq_, num_leaves_, alpha_, scale_pos_weight_
    max_depth_ = max_depth
    min_child_samples_ = min_child_samples
    min_child_weight_ = min_child_weight
    colsample_bytree_ = colsample_bytree
    subsample_ = subsample
    subsample_freq_ = subsample_freq
    num_leaves_ = num_leaves
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

modelBO = BO(one_pass, {    'max_depth': (1,10),
                            'min_child_samples': (10,600),
                            'min_child_weight': (0.001, 20),
                            'colsample_bytree': (0.1, 1),
                            'subsample': (0.5, 1),
                            'subsample_freq': (10,100),
                            'num_leaves': (10,600),
                            'alpha': (0, 20),
                            'scale_pos_weight': (1,4)
                            }, random_state=1987)
modelBO.maximize(init_points=50, n_iter=50, acq='rnd')
print modelBO.res['max']['max_params']['max_depth'],\
      modelBO.res['max']['max_params']['min_child_samples'],\
      modelBO.res['max']['max_params']['min_child_weight'],\
      modelBO.res['max']['max_params']['colsample_bytree'],\
      modelBO.res['max']['max_params']['subsample'],\
      modelBO.res['max']['max_params']['subsample_freq'],\
      modelBO.res['max']['max_params']['num_leaves'],\
      modelBO.res['max']['max_params']['alpha'],\
      modelBO.res['max']['max_params']['scale_pos_weight']
one_pass(modelBO.res['max']['max_params']['max_depth'],
         modelBO.res['max']['max_params']['min_child_samples'],
         modelBO.res['max']['max_params']['min_child_weight'],
         modelBO.res['max']['max_params']['colsample_bytree'],
         modelBO.res['max']['max_params']['subsample'],
         modelBO.res['max']['max_params']['subsample_freq'],
         modelBO.res['max']['max_params']['num_leaves'],
         modelBO.res['max']['max_params']['alpha'],
         modelBO.res['max']['max_params']['scale_pos_weight'])

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
print 'Timer lgbdt:', t1-t0


save_pred(train.id,test.id,1,pred_trn,pred_tst,'m2_lgbdt')
save_sub(test.id,pred_tst,'lgbdt2')

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

modelBO = BO(one_pass, {    'min_child_weight': (1, 20),
                            'colsample_bytree': (0.1, 1),
                            'max_depth': (1, 10),
                            'subsample': (0.5, 1),
                            'gamma': (0, 20),
                            'alpha': (0, 20),
                            'scale_pos_weight': (1,4)
                            }, random_state=201)
modelBO.maximize(init_points=50, n_iter=50, acq='rnd')
print modelBO.res['max']['max_params']['min_child_weight'],\
      modelBO.res['max']['max_params']['colsample_bytree'],\
      modelBO.res['max']['max_params']['max_depth'],\
      modelBO.res['max']['max_params']['subsample'],\
      modelBO.res['max']['max_params']['gamma'],\
      modelBO.res['max']['max_params']['alpha'],\
      modelBO.res['max']['max_params']['scale_pos_weight']
one_pass(modelBO.res['max']['max_params']['min_child_weight'],
         modelBO.res['max']['max_params']['colsample_bytree'],
         modelBO.res['max']['max_params']['max_depth'],
         modelBO.res['max']['max_params']['subsample'],
         modelBO.res['max']['max_params']['gamma'],
         modelBO.res['max']['max_params']['alpha'],
         modelBO.res['max']['max_params']['scale_pos_weight'])

#one_pass(14, 0.8071, 2, 0.6916, 14.5736, 6.23904, 3.1050)
#min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha, scale_pos_weight
#one_pass(2,          0.2025,            4,      0.8338,  17.4001, 4.0634, 2.0962)

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

modelBO = BO(one_pass, {    'depth': (2,10),
                            'reg_lambda': (0,20),
                            'feature_fraction': (0.5,1)
                            }, random_state=1987)
modelBO.maximize(init_points=50, n_iter=50, acq='rnd')
print modelBO.res['max']['max_params']['depth'],\
         modelBO.res['max']['max_params']['reg_lambda'],\
         modelBO.res['max']['max_params']['feature_fraction']
one_pass(modelBO.res['max']['max_params']['depth'],
         modelBO.res['max']['max_params']['reg_lambda'],
         modelBO.res['max']['max_params']['feature_fraction'])

print "\nDouble check Gini for full training set ", ':', eval_gini(trn_y, pred_trn)

t1 = timer()
print 'Timer catb:', t1-t0


save_pred(train.id,test.id,1,pred_trn,pred_tst,'m2_catb')
save_sub(test.id,pred_tst,'catb2')
