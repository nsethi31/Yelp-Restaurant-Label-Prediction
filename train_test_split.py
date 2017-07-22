# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:12:26 2017

@author: athir
"""

import random
import pandas as pd

data = pd.read_csv('../data/labels_cl.csv')

train_data = data[:70]
test_data = data[30:]

train_data.to_csv('../data/labels_cl_train.csv',index=0)
train_data.to_csv('../data/labels_cl_test.csv',index=0)