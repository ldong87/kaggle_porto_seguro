#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:28:44 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pk

#%% load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')

#%% show percent of missing values
trn_miss = ( (train==-1).sum(axis=0) )/train.shape[0]
tst_miss = ( (test==-1).sum(axis=0) )/test.shape[0]

#def miss_plot(data_miss, pic_name):
#    
#    plt.rcParams['figure.figsize'] = 15, 5
#    plt.figure()
#    ax = data_miss.plot(kind="bar")
#    ax.get_figure().savefig('miss_'+pic_name)
#    
#miss_plot(trn_miss, 'train')
#miss_plot(tst_miss, 'test')

#%% find relevant indices

nrow_trn = train.shape[0]
nrow_tst = test.shape[0]

with open('mis.pkl','wb') as f:
    pk.dump([nrow_trn,nrow_tst],f,protocol=pk.HIGHEST_PROTOCOL)

with open('data.pkl','wb') as f:
    pk.dump([train,test],f,protocol=pk.HIGHEST_PROTOCOL)