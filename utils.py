#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:19:28 2017

@author: ldong
"""
import numpy as np
import pandas as pd
from numba import jit
import cPickle as pk
from datetime import datetime
import glob

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def create_valid(nfold,nrow):
    np.random.seed(0)
    cv_ind = np.random.choice(nrow,size=nrow,replace=False)
    valid_ind = np.cumsum([ int(nrow/nfold) for i in xrange(nfold) ])
    valid_ind[-1] = nrow
    valid_ind = np.append(0,valid_ind)
    
    flag_valid = np.zeros([nrow,nfold],dtype=bool)
    for i in xrange(nfold):
        flag_valid[valid_ind[i]:valid_ind[i+1],i] = True
        
    return flag_valid, cv_ind

# Save validation predictions for stacking/ensembling
def save_pred(trn_ind,tst_ind,nfold,trn_pred,tst_pred,name):
    trn_val = pd.DataFrame()
    tst_val = pd.DataFrame()
    trn_val['id'] = trn_ind
    tst_val['id'] = tst_ind
    if nfold == 1:
        trn_val[name] = trn_pred
        tst_val[name] = tst_pred
    else:
        for i in xrange(nfold):
            trn_val[name+str(i)] = trn_pred[:,i]
            tst_val[name+str(i)] = tst_pred[:,i]
    with open(name+'.pkl','wb') as f:
        pk.dump([trn_val,tst_val],f,protocol=pk.HIGHEST_PROTOCOL)

def collect_meta(nfold):
    files = glob.glob('good_meta_feat/m1*.pkl')
    with open(files.pop(0),'rb') as f:
        meta_trn, meta_tst = pk.load(f)
    for ifile in files:
        with open(ifile,'rb') as f:
            meta_trn_tmp, meta_tst_tmp = pk.load(f)
        meta_trn = meta_trn.merge(meta_trn_tmp, how='left', on='id')
        meta_tst = meta_tst.merge(meta_tst_tmp, how='left', on='id')
    meta_trn_nfold = [meta_trn.filter(regex=str(ifold)+'$') for ifold in xrange(nfold)]
    meta_tst_nfold = [meta_tst.filter(regex=str(ifold)+'$') for ifold in xrange(nfold)]
    with open('meta_combo.pkl','wb') as f:
        pk.dump([meta_trn_nfold,meta_tst_nfold],f,protocol=pk.HIGHEST_PROTOCOL)

# Create submission file
def save_sub(ind,pred,name):
    sub = pd.DataFrame()
    sub['id'] = ind
    sub['target'] = pred
    sub.to_csv(name+'_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.6f')
    
def pick_feat(dt):
    train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
	"ps_car_11_cat" # Very nice spot from Tilii : https://www.kaggle.com/tilii7
    ]
    return dt.loc[:,train_features]
    
from sklearn.datasets.svmlight_format import dump_svmlight_file
import io
import re

INVALID_CHARS = re.compile(r"[\|: \n]+")
DEFAULT_NS = ''

def tovw(x, y=None, sample_weight=None):
    """Convert array or sparse matrix to Vowpal Wabbit format
    Parameters
    ----------
    x : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : {array-like}, shape (n_samples,), optional
        Target vector relative to X.
    sample_weight : {array-like}, shape (n_samples,), optional
                    sample weight vector relative to X.
    Returns
    -------
    out : {array-like}, shape (n_samples, 1)
          Training vectors in VW string format
    """

    use_truth = y is not None
    use_weight = sample_weight is not None

    # convert to numpy array if needed
    if not isinstance(x, (np.ndarray)):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # make sure this is a 2d array
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 0:
        y = y.reshape(1)

    rows, cols = x.shape

    # check for invalid characters if array has string values
    if x.dtype.char == 'S':
        for row in rows:
            for col in cols:
                x[row, col] = INVALID_CHARS.sub('.', x[row, col])

    # convert input to svmlight format
    s = io.BytesIO()
    dump_svmlight_file(x, np.zeros(rows), s)

    # parse entries to construct VW format
    rows = s.getvalue().decode('ascii').split('\n')[:-1]
    out = []
    for idx, row in enumerate(rows):
        truth = y[idx] if use_truth else 1
        weight = sample_weight[idx] if use_weight else 1
        features = row.split('0 ', 1)[1]
        # only using a single namespace and no tags
        out.append(('{y} {w} |{ns} {x}'.format(y=truth, w=weight, ns=DEFAULT_NS, x=features)))

    s.close()

    return out