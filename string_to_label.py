# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:26:21 2017

@author: athir
"""

#Function to convert the label string to one hot encoding representation

import pandas as pd

def clean_train():
    train = pd.read_csv('../data/train_labels.csv')
    for i in range(9):
        col_name = 'label_'+str(i)     
        train[col_name] = train['labels'].apply(lambda x: 1 if (str(i) in str(x)) else 0)
        
    train = train.drop(['labels'],axis=1)  
    train.to_csv('../data/train_labels_cl.csv',index=0)

    test = pd.read_csv('../data/test_labels.csv')
    for i in range(9):
        col_name = 'label_'+str(i)
        test[col_name] = test['labels'].apply(lambda x: 1 if (str(i) in str(x)) else 0)

    test = test.drop(['labels'],axis=1)
    test.to_csv('../data/test_labels_cl.csv',index=0)


    
clean_train()
