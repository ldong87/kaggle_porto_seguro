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
    
#%% xgbtree
#import model1_xgbtree as m1xgbtree
#
#def forloop(j):
#    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
#    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
#    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
#    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
#    def one_pass(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight):
#        global pred_trn
#        print 'jfold: ', j
#        pred_trn_tmp, pred_tst_tmp = m1xgbtree.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
#                                                          min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight)
#        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
#        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
#        
#        for i in xrange(nfold):
#            if i==j: 
#                continue
#            else:
#                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
#                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
#                print 'ifold: ', i
#                pred_trn_tmp, pred_tst_tmp = m1xgbtree.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
#                                                                  min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight)
#                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
#                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
#        
#        return eval_gini(trn_y, pred_trn[:,j])
#
#    modelBO = BO(one_pass, {    'min_child_weight': (1, 20),
#                                'colsample_bytree': (0.1, 1),
#                                'max_depth': (1, 10),
#                                'subsample': (0.5, 1),
#                                'gamma': (0, 20),
#                                'alpha': (0, 20),
#                                'scale_pos_weight': (1,4)
#                                }, random_state=1987)
#    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
#    
#    one_pass(modelBO.res['max']['max_params']['min_child_weight'],
#             modelBO.res['max']['max_params']['colsample_bytree'],
#             modelBO.res['max']['max_params']['max_depth'],
#             modelBO.res['max']['max_params']['subsample'],
#             modelBO.res['max']['max_params']['gamma'],
#             modelBO.res['max']['max_params']['alpha'],
#             modelBO.res['max']['max_params']['scale_pos_weight'])
#            
#    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
#
#    return pred_trn[:,j]
#
#n_core = int(nfold*10)
#pool = multiprocessing.Pool(n_core)
#t0 = timer()
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer xgbtree:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_xgbtree')

#%% xgblinear
import model1_xgblinear as m1xgblinear

def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    def one_pass(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight):
        global pred_trn
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp = m1xgblinear.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
                                                          min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp = m1xgblinear.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
                                                                  min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight)
                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

    modelBO = BO(one_pass, {    'min_child_weight': (1, 20),
                                'colsample_bytree': (0.1, 1),
                                'max_depth': (1, 10),
                                'subsample': (0.5, 1),
                                'gamma': (0, 20),
                                'alpha': (0, 20),
                                'scale_pos_weight': (1,4)
                                }, random_state=1987)
    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
    
    one_pass(modelBO.res['max']['max_params']['min_child_weight'],
             modelBO.res['max']['max_params']['colsample_bytree'],
             modelBO.res['max']['max_params']['max_depth'],
             modelBO.res['max']['max_params']['subsample'],
             modelBO.res['max']['max_params']['gamma'],
             modelBO.res['max']['max_params']['alpha'],
             modelBO.res['max']['max_params']['scale_pos_weight'])
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])

    return pred_trn[:,j]

#n_core = int(nfold*10)
#pool = multiprocessing.Pool(n_core)
#t0 = timer()
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer xgblinear:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_xgblinear')

#%% xgbdart
import model1_xgbdart as m1xgbdart

def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    def one_pass(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight,rate_drop,skip_drop):
        global pred_trn
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp = m1xgbdart.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
                                                          min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight,rate_drop,skip_drop)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp = m1xgbdart.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
                                                                  min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,scale_pos_weight,rate_drop,skip_drop)
                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

    modelBO = BO(one_pass, {    'min_child_weight': (1, 20),
                                'colsample_bytree': (0.1, 1),
                                'max_depth': (1, 10),
                                'subsample': (0.5, 1),
                                'gamma': (0, 20),
                                'alpha': (0, 20),
                                'scale_pos_weight': (1,4),
                                'rate_drop': (0,0.5),
                                'skip_drop': (0.5,1)
                                }, random_state=1987)
    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
    
    one_pass(modelBO.res['max']['max_params']['min_child_weight'],
             modelBO.res['max']['max_params']['colsample_bytree'],
             modelBO.res['max']['max_params']['max_depth'],
             modelBO.res['max']['max_params']['subsample'],
             modelBO.res['max']['max_params']['gamma'],
             modelBO.res['max']['max_params']['alpha'],
             modelBO.res['max']['max_params']['scale_pos_weight'],
             modelBO.res['max']['max_params']['rate_drop'],
             modelBO.res['max']['max_params']['skip_drop'])
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])

    return pred_trn[:,j]

#n_core = int(nfold*10)
#pool = multiprocessing.Pool(n_core)
#t0 = timer()
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer xgbdart:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))

save_pred(train.id,nfold,pred_trn,'m1_xgbdart')

#%% lgbdt
#import model1_lgbdt as m1lgbdt
#
#def forloop(j):
#    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
#    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
#    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
#    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
#    def one_pass( max_depth,
#                  min_child_samples,
#                  min_child_weight,
#                  colsample_bytree,
#                  subsample,
#                  subsample_freq,
#                  num_leaves,
#                  alpha,
#                  scale_pos_weight):
#        global pred_trn
#        print 'jfold: ', j
#        pred_trn_tmp, pred_tst_tmp = m1lgbdt.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
#                                                          max_depth,
#                                                          min_child_samples,
#                                                          min_child_weight,
#                                                          colsample_bytree,
#                                                          subsample,
#                                                          subsample_freq,
#                                                          num_leaves,
#                                                          alpha,
#                                                          scale_pos_weight)
#        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
#        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
#        
#        for i in xrange(nfold):
#            if i==j: 
#                continue
#            else:
#                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
#                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
#                print 'ifold: ', i
#                pred_trn_tmp, pred_tst_tmp = m1lgbdt.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
#                                                                  max_depth,
#                                                                  min_child_samples,
#                                                                  min_child_weight,
#                                                                  colsample_bytree,
#                                                                  subsample,
#                                                                  subsample_freq,
#                                                                  num_leaves,
#                                                                  alpha,
#                                                                  scale_pos_weight)
#                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
#                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
#        
#        return eval_gini(trn_y, pred_trn[:,j])
#
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
#            
#    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
#    
#    return pred_trn[:,j]
#
#n_core = int(nfold*10)
#t0 = timer()
#pool = multiprocessing.Pool(n_core)
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer lgbdt:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_lgbdt')

#%% lgbdart
import model1_lgbdart as m1lgbdart

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
        global pred_trn
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp = m1lgbdart.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
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
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp = m1lgbdart.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
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
        
        return eval_gini(trn_y, pred_trn[:,j])

    modelBO = BO(one_pass, {    'max_depth': (1,10),
                                'min_child_samples': (10,600),
                                'min_child_weight': (0.001, 20),
                                'colsample_bytree': (0.1, 1),
                                'subsample': (0.5, 1),
                                'subsample_freq': (10,100),
                                'num_leaves': (10,600),
                                'alpha': (0, 20),
                                'scale_pos_weight': (1,4),
                                'drop_rate': (0,0.5),
                                'skip_drop': (0.5,1)
                                }, random_state=1987)
    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
    
    one_pass(modelBO.res['max']['max_params']['max_depth'],
             modelBO.res['max']['max_params']['min_child_samples'],
             modelBO.res['max']['max_params']['min_child_weight'],
             modelBO.res['max']['max_params']['colsample_bytree'],
             modelBO.res['max']['max_params']['subsample'],
             modelBO.res['max']['max_params']['subsample_freq'],
             modelBO.res['max']['max_params']['num_leaves'],
             modelBO.res['max']['max_params']['alpha'],
             modelBO.res['max']['max_params']['scale_pos_weight'],
             modelBO.res['max']['max_params']['drop_rate'],
             modelBO.res['max']['max_params']['skip_drop'])
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return pred_trn[:,j]

#n_core = int(nfold*10)
#t0 = timer()
#pool = multiprocessing.Pool(n_core)
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer lgbdart:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_lgbdart')

#%% lgbrf
import model1_lgbrf as m1lgbrf

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
        global pred_trn
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp = m1lgbrf.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
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
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp = m1lgbrf.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
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
        
        return eval_gini(trn_y, pred_trn[:,j])

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
    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
    
    one_pass(modelBO.res['max']['max_params']['max_depth'],
             modelBO.res['max']['max_params']['min_child_samples'],
             modelBO.res['max']['max_params']['min_child_weight'],
             modelBO.res['max']['max_params']['colsample_bytree'],
             modelBO.res['max']['max_params']['subsample'],
             modelBO.res['max']['max_params']['subsample_freq'],
             modelBO.res['max']['max_params']['num_leaves'],
             modelBO.res['max']['max_params']['alpha'],
             modelBO.res['max']['max_params']['scale_pos_weight'])
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return pred_trn[:,j]

#n_core = int(nfold*10)
#t0 = timer()
#pool = multiprocessing.Pool(n_core)
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer lgbrf:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_lgbrf')

#%% catb
#import model1_catb as m1catb
#
#def forloop(j):
#    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
#    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
#    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
#    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
#    def one_pass( depth,
#                  reg_lambda,
#                  feature_fraction):
#        global pred_trn
#        print 'jfold: ', j
#        pred_trn_tmp, pred_tst_tmp = m1catb.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x,
#                                                          depth,
#                                                          reg_lambda,
#                                                          feature_fraction)
#        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
#        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
#        
#        for i in xrange(nfold):
#            if i==j: 
#                continue
#            else:
#                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
#                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
#                print 'ifold: ', i
#                pred_trn_tmp, pred_tst_tmp = m1catb.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x,
#                                                                  depth,
#                                                                  reg_lambda,
#                                                                  feature_fraction)
#                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
#                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
#        
#        return eval_gini(trn_y, pred_trn[:,j])
#
#    modelBO = BO(one_pass, {    'depth': (2,10),
#                                'reg_lambda': (0,20),
#                                'feature_fraction': (0.5,1)
#                                }, random_state=1987)
#    modelBO.maximize(init_points=5, n_iter=5, acq='rnd')
#    
#    one_pass(modelBO.res['max']['max_params']['depth'],
#             modelBO.res['max']['max_params']['reg_lambda'],
#             modelBO.res['max']['max_params']['feature_fraction'])
##    one_pass(6,14.0,1.0)
#            
#    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
#    
#    return pred_trn[:,j]
##forloop(4)
#n_core = int(nfold*10)
#pool = multiprocessing.Pool(n_core)
#t0 = timer()
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer catb:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_catb')

#%% sk nn

#%% knn
import model1_knn as m1knn

def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    def one_pass():
        global pred_trn
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp = m1knn.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        
        for i in xrange(nfold):
            if i==j: 
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                val_tmp_x_ = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
                val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
                print 'ifold: ', i
                pred_trn_tmp, pred_tst_tmp = m1knn.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x)
                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

    one_pass()
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return pred_trn[:,j]

n_core = int(nfold)
pool = multiprocessing.Pool(n_core)
t0 = timer()
pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
t1 = timer()
print 'Timer knn:', t1-t0

print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))

save_pred(train.id,nfold,pred_trn,'m1_knn')

#%% elastic net

#%% svm

#%% vw
import model1_vw as m1vw
#from bayes_opt_test import BayesianOptimization as BO

def dump2vw(trn_x,trn_y):
    data_vw = []
    data_vw_ = []
    for i in xrange(nfold):
        trn_tmp_x = trn_x.loc[cv_ind[~flag_valid[:,i]]]
        trn_tmp_y = trn_y.loc[cv_ind[~flag_valid[:,i]]]
        val_tmp_x = trn_x.loc[cv_ind[ flag_valid[:,i]]]
        trn_tmp_y.loc[trn_tmp_y==0] = -1
    
        trn_tmp_vw = tovw(trn_tmp_x,trn_tmp_y)
        val_tmp_vw = tovw(val_tmp_x)
        
        data_vw.append([trn_tmp_vw,val_tmp_vw])
        
        data_vw_tmp = []
        for j in xrange(nfold):
            if j == i:
                continue
            else:
                trn_tmp_x_ = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_ = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
                trn_tmp_y_[trn_tmp_y_==0]=-1
                
                trn_tmp_vw_ = tovw(trn_tmp_x_,trn_tmp_y_)
                
                data_vw_tmp.append([trn_tmp_vw_,j])
        data_vw_.append(data_vw_tmp)
        
    with open('vw_data_'+str(nfold)+'fold.pkl','wb') as f:
        pk.dump([data_vw,data_vw_],f,protocol=pk.HIGHEST_PROTOCOL)

# only use this when nfold changes
dump2vw(trn_x,trn_y)        

with open('vw_data_'+str(nfold)+'fold.pkl','rb') as f:
        vw_data, vw_data_ = pk.load(f)

def forloop(j):
    trn_tmp_xy = vw_data[j][0]
    val_tmp_x = vw_data[j][1]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]

    def one_pass():
        global pred_trn
        print 'jfold: ', j
        pred_trn_tmp = m1vw.model_pred(trn_tmp_xy,val_tmp_x,val_tmp_y)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        
        vw_data_tmp = vw_data_[j]
        for i in xrange(nfold-1):
            trn_tmp_xy_ = vw_data_tmp[0]
            val_tmp_x_ = vw_data[vw_data_tmp[1]][1]
            val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,vw_data_tmp[1]]]]
 
            print 'ifold: ', i
            pred_trn_tmp, pred_tst_tmp = m1vw.model_pred(trn_tmp_xy_,val_tmp_x_,val_tmp_y_)
            print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
            pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

#    modelBO = BO(one_pass, {    'passes': (1,6),
#                                'l1': (0,20),
#                                'l2': (0,20)
#                                }, random_state=1987)
#    modelBO.maximize(init_points=1, n_iter=1, acq='rnd')
#    
#    one_pass(modelBO.res['max']['max_params']['passes'],
#             modelBO.res['max']['max_params']['l1'],
#             modelBO.res['max']['max_params']['l2'])
    one_pass()
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return pred_trn[:,j]

#n_core = int(nfold*10)
#pool = multiprocessing.Pool(n_core)
#t0 = timer()
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer vw:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_vw')

#%% vwnn
import model1_vwnn as m1vwnn
#from bayes_opt_test import BayesianOptimization as BO       

#with open('vw_data_'+str(nfold)+'fold.pkl','rb') as f:
#        vw_data, vw_data_ = pk.load(f)

def forloop(j):
    trn_tmp_xy = vw_data[j][0]
    val_tmp_x = vw_data[j][1]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]

    def one_pass():
        global pred_trn
        print 'jfold: ', j
        pred_trn_tmp = m1vwnn.model_pred(trn_tmp_xy,val_tmp_x,val_tmp_y)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        
        vw_data_tmp = vw_data_[j]
        for i in xrange(nfold-1):
            trn_tmp_xy_ = vw_data_tmp[0]
            val_tmp_x_ = vw_data[vw_data_tmp[1]][1]
            val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,vw_data_tmp[1]]]]
 
            print 'ifold: ', i
            pred_trn_tmp, pred_tst_tmp = m1vwnn.model_pred(trn_tmp_xy_,val_tmp_x_,val_tmp_y_)
            print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
            pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])

#    modelBO = BO(one_pass, {    'passes': (1,6),
#                                'l1': (0,20),
#                                'l2': (0,20)
#                                }, random_state=1987)
#    modelBO.maximize(init_points=1, n_iter=1, acq='rnd')
#    
#    one_pass(modelBO.res['max']['max_params']['passes'],
#             modelBO.res['max']['max_params']['l1'],
#             modelBO.res['max']['max_params']['l2'])
    one_pass()
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return pred_trn[:,j]

#n_core = int(nfold*10)
#pool = multiprocessing.Pool(n_core)
#t0 = timer()
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
#t1 = timer()
#print 'Timer vwnn:', t1-t0
#
#print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#
#save_pred(train.id,nfold,pred_trn,'m1_vwnn')

#%% gp

#%% bayes

#%% libffm

#%% rgf

#%% topological data analysis