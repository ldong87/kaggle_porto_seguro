#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:31:15 2017

@author: ldong
"""

import numpy as np
import pandas as pd
import cPickle as pk
from utils import *

with open('m2_xgbtree.pkl','rb') as f:
    m2xgb_trn, m2xgb_tst = pk.load(f)
    
with open('m2_lgbdt.pkl','rb') as f:
    m2lgb_trn, m2lgb_tst = pk.load(f)
    
with open('m2_catb.pkl','rb') as f:
    m2catb_trn, m2catb_tst = pk.load(f)
    
with open('data.pkl','rb') as f:
        [train, test] = pk.load(f)
        
gini_xgb = eval_gini(train.target,m2xgb_trn.m2_xgbtree)
gini_lgb = eval_gini(train.target,m2lgb_trn.m2_lgbdt)
gini_catb = eval_gini(train.target,m2catb_trn.m2_catb)
print 'Gini of xgbtree: ', gini_xgb
print 'Gini of lgbdt: ',   gini_lgb
print 'Gini of catb: ',    gini_catb

save_sub(test.id,m2xgb_tst.m2_xgbtree,'xgbfinal')
save_sub(test.id,m2lgb_tst.m2_lgbdt,'lgbfinal')
save_sub(test.id,m2catb_tst.m2_catb,'catbfinal')


gini_sum = gini_xgb + gini_lgb + gini_catb
mix_trn = (gini_xgb/gini_sum)*m2xgb_trn.m2_xgbtree + (gini_lgb/gini_sum)*m2lgb_trn.m2_lgbdt + (gini_catb/gini_sum)*m2catb_trn.m2_catb

print 'Gini of mix: ',    eval_gini(train.target,mix_trn)

mix_tst = (gini_xgb/gini_sum)*m2xgb_tst.m2_xgbtree + (gini_lgb/gini_sum)*m2lgb_tst.m2_lgbdt + (gini_catb/gini_sum)*m2catb_tst.m2_catb

save_sub(test.id,mix_tst,'mix')
