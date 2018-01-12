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

#%% vw
import model1_vw as m1vw

#with open('vw_tst_.pkl','wb') as f:
#    pk.dump(tovw(tst_x),f,protocol=pk.HIGHEST_PROTOCOL)
    
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
#dump2vw(trn_x,trn_y)        

with open('vw_data_'+str(nfold)+'fold.pkl','rb') as f:
        vw_data, vw_data_ = pk.load(f)
with open('vw_tst_.pkl','rb') as f:
        tst_x = pk.load(f)

best_iter = 0
pred_tst = np.zeros([nrow_tst,nfold])
def forloop(j):
    trn_tmp_xy = vw_data[j][0]
    val_tmp_x = vw_data[j][1]
    val_tmp_y = trn_y.iloc[cv_ind[ flag_valid[:,j]]]

    def one_pass():
        global pred_trn, pred_tst
        pred_tst[:,j] = 0
        print 'jfold: ', j
        pred_trn_tmp, pred_tst_tmp, best_iter = m1vw.model_pred(trn_tmp_xy,val_tmp_x,val_tmp_y, tst_x)
        print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
        pred_trn[cv_ind[flag_valid[:,j]],j] = pred_trn_tmp
        pred_tst[:,j] = pred_tst_tmp
        
        vw_data_tmp = vw_data_[j]
        for i in xrange(nfold-1):
            trn_tmp_xy_ = vw_data_tmp[i][0]
            val_tmp_x_ = vw_data[vw_data_tmp[i][1]][1]
            val_tmp_y_ = trn_y.iloc[cv_ind[ flag_valid[:,vw_data_tmp[i][1]]]]
 
            print 'ifold: ', i
            pred_trn_tmp, pred_tst_tmp, best_iter  = m1vw.model_pred(trn_tmp_xy_,val_tmp_x_,val_tmp_y_, tst_x)
            print 'double check gini of ifold=', i,' pred = ', eval_gini(val_tmp_y_, pred_trn_tmp)
            pred_trn[cv_ind[flag_valid[:,vw_data_tmp[i][1]]],j] = pred_trn_tmp
            pred_tst[:,j] = pred_tst[:,j] + pred_tst_tmp
        
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
    
    return [pred_trn[:,j], pred_tst[:,j]/float(nfold)]

n_core = int(nfold)
pool = multiprocessing.Pool(n_core)
t0 = timer()
result_combo = pool.map(forloop, range(nfold))   
pred_trn = np.transpose(np.array([result_combo[i][0] for i in xrange(nfold)]))
pred_tst = np.transpose(np.array([result_combo[i][1] for i in xrange(nfold)]))
#pred_trn = np.transpose(np.array(pool.map(forloop, range(nfold))))
t1 = timer()
print 'Timer vw:', t1-t0

print "\nGini for full training set avg:" , eval_gini(trn_y, np.mean(pred_trn,axis=1))

save_pred(train.id,test.id,nfold,pred_trn,pred_tst,'m1_vw')

