#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:53:30 2017

@author: ldong
"""

import numpy as np
import pandas as pd
from vowpalwabbit import pyvw
from utils import *

def model_pred(trn_tmp_xy,val_tmp_x,val_tmp_y):

    param = ['-b 7 ' +
             '--link logistic ' +
            '--loss_function logistic '  +
            '-l 0.2 ' +
            '--l1 0 ' +
            '--l2 0 ' +
            '--nn 70 ' +
            '--dropout ' +
            '--holdout_off ' +
            '--total 32 ' +
            '-f vw.model ' +
            '--readable_model vw.readable.model']
    vw = pyvw.vw(*param)
    for iteration in xrange(400):
        for i in xrange(len(trn_tmp_xy)):
            vw.learn(trn_tmp_xy[i])
    vw.finish()
    vw = pyvw.vw("-i vw.model -t")
    pred_trn_tmp = [vw.predict(sample) for sample in val_tmp_x]

    return pred_trn_tmp 



param = ['-b 7 ' +
         '--link logistic ' +
        '--loss_function logistic '  +
        '-l 0.2 ' +
        '--l1 0 ' +
        '--l2 0 ' +
        '--nn 10 ' +
        '--dropout ' +
        '--holdout_off ' +
        '--total 32 ' +
        '-f vw.model ' +
        '--readable_model vw.readable.model']
vw = pyvw.vw(*param)
for iteration in xrange(200):
    for i in xrange(len(trn_tmp_xy)):
        vw.learn(trn_tmp_xy[i])
vw.finish()
vw = pyvw.vw("-i vw.model -t")
pred_trn_tmp = [vw.predict(sample) for sample in val_tmp_x]
print 'double check gini of jfold=', j, ' pred = ', eval_gini(val_tmp_y, pred_trn_tmp)
