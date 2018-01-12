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

with open('mis.pkl','rb') as f:
    nrow_trn,nrow_tst = pk.load(f)

nfold = 5 # must be greater than 2
with open('data.pkl','rb') as f:
        [train, test] = pk.load(f)
        
trn_x = train.loc[:,'ps_ind_01':]
trn_y = train.target
tst_x = test.loc[:,'ps_ind_01':]
pred_trn = np.zeros([nrow_trn,nfold])

flag_valid, cv_ind = create_valid(nfold, nrow_trn)
    
#%% lgbdt
import model1_lgbdt as m1lgbdt
best_iter = 0
pred_tst = np.zeros([nrow_tst,nfold])
def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    def one_pass( max_depth,
                  min_child_samples,
                  min_child_weight,
                  colsample_bytree,
                  subsample,
                  subsample_freq,
                  num_leaves,
                  alpha,
                  scale_pos_weight):
        global pred_trn, pred_tst
        pred_tst[:,j] = 0
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp, best_iter = m1lgbdt.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
                                                          max_depth,
                                                          min_child_samples,
                                                          min_child_weight,
                                                          colsample_bytree,
                                                          subsample,
                                                          subsample_freq,
                                                          num_leaves,
                                                          alpha,
                                                          scale_pos_weight)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        pred_tst[:,j] = pred_tst_tmp
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp, best_iter = m1lgbdt.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
                                                                  max_depth,
                                                                  min_child_samples,
                                                                  min_child_weight,
                                                                  colsample_bytree,
                                                                  subsample,
                                                                  subsample_freq,
                                                                  num_leaves,
                                                                  alpha,
                                                                  scale_pos_weight)
                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
                pred_tst[:,j] = pred_tst[:,j] + pred_tst_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

#    modelBO = BO(one_pass, {    'max_depth': (1,10),
#                                'min_child_samples': (10,600),
#                                'min_child_weight': (0.001, 20),
#                                'colsample_bytree': (0.1, 1),
#                                'subsample': (0.5, 1),
#                                'subsample_freq': (10,100),
#                                'num_leaves': (10,600),
#                                'alpha': (0, 20),
#                                'scale_pos_weight': (1,4)
#                                }, random_state=1987)
#    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
#    
#    one_pass(modelBO.res['max']['max_params']['max_depth'],
#             modelBO.res['max']['max_params']['min_child_samples'],
#             modelBO.res['max']['max_params']['min_child_weight'],
#             modelBO.res['max']['max_params']['colsample_bytree'],
#             modelBO.res['max']['max_params']['subsample'],
#             modelBO.res['max']['max_params']['subsample_freq'],
#             modelBO.res['max']['max_params']['num_leaves'],
#             modelBO.res['max']['max_params']['alpha'],
#             modelBO.res['max']['max_params']['scale_pos_weight'])
  
    one_pass(5,600,6.4359,0.2295,0.8546,36,430,11.0516,2.6977)
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return [pred_trn[:,j], pred_tst[:,j]/float(nfold)]

n_core = int(nfold*10)
pool = multiprocessing.Pool(n_core)
t0 = timer()
result_combo = pool.map(forloop, range(nfold))   
pred_trn = np.transpose(np.array([result_combo[i][0] for i in xrange(nfold)]))
pred_tst = np.transpose(np.array([result_combo[i][1] for i in xrange(nfold)]))
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
t1 = timer()
print 'Timer lgbdt:', t1-t0

print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))

save_pred(train.id,test.id,nfold,pred_trn,pred_tst,'m1_lgbdt')

#%% lgbdart
import model1_lgbdart as m1lgbdart
best_iter = 0
pred_tst = np.zeros([nrow_tst,nfold])
def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    def one_pass( max_depth,
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
        global pred_trn, pred_tst
        pred_tst[:,j] = 0
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp, best_iter = m1lgbdart.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
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
                                                          skip_drop)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        pred_tst[:,j] = pred_tst_tmp
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp, best_iter = m1lgbdart.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
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
                                                                  skip_drop)
                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
                pred_tst[:,j] = pred_tst[:,j] + pred_tst_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

#    modelBO = BO(one_pass, {    'max_depth': (1,10),
#                                'min_child_samples': (10,600),
#                                'min_child_weight': (0.001, 20),
#                                'colsample_bytree': (0.1, 1),
#                                'subsample': (0.5, 1),
#                                'subsample_freq': (10,100),
#                                'num_leaves': (10,600),
#                                'alpha': (0, 20),
#                                'scale_pos_weight': (1,4),
#                                'drop_rate': (0,0.5),
#                                'skip_drop': (0.5,1)
#                                }, random_state=1987)
#    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
#    
#    one_pass(modelBO.res['max']['max_params']['max_depth'],
#             modelBO.res['max']['max_params']['min_child_samples'],
#             modelBO.res['max']['max_params']['min_child_weight'],
#             modelBO.res['max']['max_params']['colsample_bytree'],
#             modelBO.res['max']['max_params']['subsample'],
#             modelBO.res['max']['max_params']['subsample_freq'],
#             modelBO.res['max']['max_params']['num_leaves'],
#             modelBO.res['max']['max_params']['alpha'],
#             modelBO.res['max']['max_params']['scale_pos_weight'],
#             modelBO.res['max']['max_params']['drop_rate'],
#             modelBO.res['max']['max_params']['skip_drop'])
       
    one_pass(9,305,2.0337,0.8955,0.6832,17,292,6.6038,2.9325,0.1807,0.9160)
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return [pred_trn[:,j], pred_tst[:,j]/float(nfold)]

n_core = int(nfold*10)
pool = multiprocessing.Pool(n_core)
t0 = timer()
result_combo = pool.map(forloop, range(nfold))   
pred_trn = np.transpose(np.array([result_combo[i][0] for i in xrange(nfold)]))
pred_tst = np.transpose(np.array([result_combo[i][1] for i in xrange(nfold)]))
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
t1 = timer()
print 'Timer lgbdart:', t1-t0

print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))

save_pred(train.id,test.id,nfold,pred_trn,pred_tst,'m1_lgbdart')

#%% lgbrf
import model1_lgbrf as m1lgbrf
best_iter = 0
pred_tst = np.zeros([nrow_tst,nfold])
def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    def one_pass( max_depth,
                  min_child_samples,
                  min_child_weight,
                  colsample_bytree,
                  subsample,
                  subsample_freq,
                  num_leaves,
                  alpha,
                  scale_pos_weight):
        global pred_trn, pred_tst
        pred_tst[:,j] = 0
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp, best_iter = m1lgbrf.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
                                                          max_depth,
                                                          min_child_samples,
                                                          min_child_weight,
                                                          colsample_bytree,
                                                          subsample,
                                                          subsample_freq,
                                                          num_leaves,
                                                          alpha,
                                                          scale_pos_weight)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        pred_tst[:,j] = pred_tst_tmp
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp, best_iter = m1lgbrf.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
                                                                  max_depth,
                                                                  min_child_samples,
                                                                  min_child_weight,
                                                                  colsample_bytree,
                                                                  subsample,
                                                                  subsample_freq,
                                                                  num_leaves,
                                                                  alpha,
                                                                  scale_pos_weight)
                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
                pred_tst[:,j] = pred_tst[:,j] + pred_tst_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

#    modelBO = BO(one_pass, {    'max_depth': (1,10),
#                                'min_child_samples': (10,600),
#                                'min_child_weight': (0.001, 20),
#                                'colsample_bytree': (0.1, 1),
#                                'subsample': (0.5, 1),
#                                'subsample_freq': (10,100),
#                                'num_leaves': (10,600),
#                                'alpha': (0, 20),
#                                'scale_pos_weight': (1,4)
#                                }, random_state=1987)
#    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
#    
#    one_pass(modelBO.res['max']['max_params']['max_depth'],
#             modelBO.res['max']['max_params']['min_child_samples'],
#             modelBO.res['max']['max_params']['min_child_weight'],
#             modelBO.res['max']['max_params']['colsample_bytree'],
#             modelBO.res['max']['max_params']['subsample'],
#             modelBO.res['max']['max_params']['subsample_freq'],
#             modelBO.res['max']['max_params']['num_leaves'],
#             modelBO.res['max']['max_params']['alpha'],
#             modelBO.res['max']['max_params']['scale_pos_weight'])
    one_pass(4,292,15.6113,0.2682,0.6202,54,171,3.706,2.2753)
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return [pred_trn[:,j], pred_tst[:,j]/float(nfold)]

n_core = int(nfold*10)
pool = multiprocessing.Pool(n_core)
t0 = timer()
result_combo = pool.map(forloop, range(nfold))   
pred_trn = np.transpose(np.array([result_combo[i][0] for i in xrange(nfold)]))
pred_tst = np.transpose(np.array([result_combo[i][1] for i in xrange(nfold)]))
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
t1 = timer()
print 'Timer lgbrf:', t1-t0

print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))

save_pred(train.id,test.id,nfold,pred_trn,pred_tst,'m1_lgbrf')

