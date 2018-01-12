#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:20:24 2017

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

#%% Bernoulli NB
import model1_nb as m1nb
best_iter = 0
pred_tst = np.zeros([nrow_tst,nfold])
def forloop(j):
    trn_tmp_x = trn_x.iloc[cv_ind[~flag_valid[:,j]]]
    trn_tmp_y = trn_y.iloc[cv_ind[~flag_valid[:,j]]]
    val_tmp_x = trn_x.iloc[cv_ind[ flag_valid[:,j]]]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]
    def one_pass():
        global pred_trn, pred_tst
        pred_tst[:,j] = 0
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp, best_iter = m1nb.model_pred(trn_tmp_x,trn_tmp_y,val_tmp_x,val_tmp_y,tst_x)
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
                pred_trn_tmp, pred_tst_tmp, best_iter = m1nb.model_pred(trn_tmp_x_,trn_tmp_y_,val_tmp_x_,val_tmp_y_,tst_x)
                print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
                pred_trn[cv_ind[flag_valid[:,i]],j] = pred_trn_tmp
                pred_tst[:,j] = pred_tst[:,j] + pred_tst_tmp
        
        return eval_gini(trn_y, pred_trn[:,j])
    
#    modelBO = BO(one_pass, {    'lr': (0.1,1.5)
#                                }, random_state=1987)
#    modelBO.explore({'lr': [1.0]})
#    modelBO.maximize(init_points=10, n_iter=20, acq='rnd')
    
#    one_pass(modelBO.res['max']['max_params']['lr']
#             )
    one_pass()
            
    print "\nGini for full training set ", j, ':', eval_gini(trn_y, pred_trn[:,j])
    
    return [pred_trn[:,j], pred_tst[:,j]/float(nfold)]

n_core = int(nfold)
pool = multiprocessing.Pool(n_core)
t0 = timer()
result_combo = pool.map(forloop, range(nfold))   
pred_trn = np.transpose(np.array([result_combo[i][0] for i in xrange(nfold)]))
pred_tst = np.transpose(np.array([result_combo[i][1] for i in xrange(nfold)]))
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
t1 = timer()
print 'Timer nb:', t1-t0

print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))

save_pred(train.id,test.id,nfold,pred_trn,pred_tst,'m1_nb')

