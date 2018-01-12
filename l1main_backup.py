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
from bayes_opt import BayesianOptimization as BO


np.random.seed(0)

with open('mis.pkl','rb') as f:
    nrow_trn,nrow_tst = pk.load(f)

nfold = 3 # must be greater than 2
with open('data.pkl','rb') as f:
        [train, test] = pk.load(f)
        
trn_x = train.loc[:,'ps_ind_01':]
trn_y = train.target
tst_x = test.loc[:,'ps_ind_01':]

flag_valid, cv_ind = create_valid(nfold, nrow_trn)
    
pred_trn = np.zeros([nrow_trn,nfold])
pred_tst = np.zeros(nrow_tst)

#%% xgbtree
import model1_xgbtree as m1xgbtree

def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    print 'jfold: ', j
    pred_trn_tmp, pred_tst_tmp = m1xgbtree.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x)
    print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
    pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
    
    for i in xrange(nfold):
        if i==j: 
            continue
        else:
            trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
            trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
            val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
            val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
            print 'ifold: ', i
            pred_trn_tmp, pred_tst_tmp = m1xgbtree.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x)
            print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
            pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return pred_trn[:,j]

pool = multiprocessing.Pool(48)
pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))


#for j in xrange(nfold):
#    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
#    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
#    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
#    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
#    print 'jfold: ', j
#    pred_trn_tmp, pred_tst_tmp = m1xgbtree.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x)
#    print 'double check gini of jfold pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
#    pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
#    pred_tst += pred_tst_tmp
#    
#    for i in xrange(nfold):
#        if i==j: 
#            continue
#        else:
#            trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#            trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,i] & ~flag_valid[:,j]]]
#            val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,i]]]
#            val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,i]]]
#            print 'ifold: ', i
#            pred_trn_tmp, pred_tst_tmp = m1xgbtree.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x)
#            print 'double check gini of ifold pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
#            pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
#            pred_tst += pred_tst_tmp
#            
#    print "\nGini for full training set j:" , eval_gini(trn_y, pred_trn[:,j])
    
print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))
#pred_tst /= nfold*nfold  # Average test set predictions

save_pred(train.id,nfold,pred_trn,'m1_xgbtree')
#save_sub(test.id,pred_tst,'m1_xgbtree')

#%% xgblinear

#%% xgbdart

#%% lgbdt

#%% lgbrf

#%% lgbdart

#%% catb

#%% sknn

#%% knn

#%% enet

#%% svm

#%% vw

#%% vwnn

#%% gp

#%% bayes