#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:22:04 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import cPickle as pk

with open('data.pkl','rb') as f:
    train,test = pk.load(f)
    
np.random.seed(0)
nfold = 5
nrow = train.shape[0] # num of data points

cv_ind = np.random.choice(nrow,size=nrow,replace=False)
valid_ind = np.cumsum([ int(nrow/nfold) for i in xrange(nfold) ])
valid_ind[-1] = nrow
valid_ind = np.append(0,valid_ind)

flag_valid = np.zeros([nrow,nfold],dtype=bool)
for i in xrange(nfold):
    flag_valid[valid_ind[i]:valid_ind[i+1],i] = True