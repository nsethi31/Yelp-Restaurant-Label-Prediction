# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:14:08 2017

@author: athir
"""

import numpy as np

import pandas as pd

import xgboost as xgb

import time

'''
Given a numpy array of probabilities, this function returns classification array (of either 0 or 1)
The thresholding is performed using the parameter thresh
if prob>thresh return 1, else return 0
'''
def np_thresh(x1,thresh):
    
    x1 = x1-thresh
    
    x1 = x1>0
    
    x1 = np.array(x1,dtype=int)
    
    return x1
    
'''
Utility function to calculate precision,recall,fscore
'''
def fscore(x1,x2):

    tru_pos  = np.sum(np.array((np.logical_and(x1,x2)),dtype=int))

    actual_pos = np.sum(x1)
    
    pred_pos = np.sum(x2)
    if(pred_pos>0):
        precision = float(tru_pos)/pred_pos
    else:
        precision = 0
    
    if(actual_pos>0):
        recall = float(tru_pos)/actual_pos
    else:
        recall=0
    
    
    if(precision+recall>0):
        fscore = (2*precision*recall)/(precision+recall)
    else:
        fscore = 0
    
      
    
    return precision,recall,fscore
	
start_time = time.time()
	
'''
Part1:Cleaning
After pretraining, we have two files feat_holder.npy which is a num_unique_train_photosx2048 numpy array
and feat_holder_test.npy which is a num_unique_test_photosx2048 numpy array
Since each business has multiple instances (photos), we need some way to
condense this multiple instance information to a single instance information
The simplest way to do this is to take the mean of all the instances per business, and assign it
as a feature vector for that business.
At the end of this, we have a num_unique_businessx2048 array which we will use as a festure vector for training
The result of this part of code is a pandas dataframe 
'''

train_to_biz = pd.read_csv('../data/train_id.csv')

train_image_features = np.load('../data/train_inception_7.npy',mmap_mode='r')

uni_bus = train_to_biz['business_id'].unique()

coll_arr = np.zeros([len(uni_bus),2048])

for nb,ub in enumerate(uni_bus):
    tbz = np.array(train_to_biz['business_id']==ub,dtype=bool)
    x1 = np.array(train_image_features[tbz,:])
    x1 = np.mean(x1,axis=0)
    x1 = x1.reshape([1,2048])
    coll_arr[nb,:2048] = x1
	
	
biz_features = pd.DataFrame(uni_bus,columns=['business_id'])

coll_arr = pd.DataFrame(coll_arr)

frames = [biz_features,coll_arr]

biz_features = pd.concat(frames,axis=1)

del train_to_biz,train_image_features,coll_arr,frames

print('time taken for Part1:Cleaning',time.time()-start_time)


'''
Part2: Estimating the performance using cross validation
We will use 5 fold cross validation to estimate the performance of our model,
and in the process tune the parameters of the model
Here we will be using the xgboost package
This part of code will also serve a second purpose
If we would like to use this model in an ensemble at a later stage, we will save the results
from cross validation, and use them as features in the ensembling stage
Since xgboost doesnt have a straightforward way of training multi label classification problems,
we will build 9 different binary classification models. However, this does not take into account the 
realtionship between the different labels, and might not result in the best performance
We will overcome this deficiency in the ensembling stage by using a slightly different architecture
To calculate the fscore, we will need to use some threshold to convert the probabilites into binary labels
The thresholding step is not very critical if our goal is to use this model only as features for the second level ensemble
But if this model is the final model, then the thresholding parameter also needs to be tuned.
Here, I've used 0.48 as the threshold
'''


cv = 1

submit = 1

num_cv = 5

from sklearn.cross_validation import KFold

skf = list(KFold(biz_features.shape[0],num_cv,random_state=42))

dataset_blend_train = np.zeros([biz_features.shape[0],9])

labels = ['label_'+str(i) for i in range(9)]

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 3
param['subsample'] = 0.6
param['silent'] = 1
param['nthread'] = 4
param['eval_metric'] = "mlogloss"
num_round = 100
param['num_class'] = 2


iter_label = {}


for nb,lb in enumerate(labels):
    iter_n = 0
    train_cl = pd.read_csv('../data/train_labels_cl.csv')

    train_cl = dict(np.array(train_cl[['business_id',lb]]))

    biz_features['lb'] = biz_features['business_id'].apply(lambda x: train_cl[x])
    if(cv):
        for (training,testing) in skf:
        
            df_train = biz_features.iloc[training]
            
            df_test = biz_features.iloc[testing]

            df_train_values = np.array(df_train['lb'],dtype=bool)
            
            df_train_features = df_train.drop(['business_id','lb'],axis=1)
            
            df_test_values = np.array(df_test['lb'],dtype=bool)
            
            df_test_features = df_test.drop(['business_id','lb'],axis=1)

            xg_train = xgb.DMatrix(df_train_features, label=df_train_values)
            
            xg_test = xgb.DMatrix(df_test_features, label=df_test_values)
            
            watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
            
            bst = xgb.train(param, xg_train, num_round,watchlist,early_stopping_rounds=4,verbose_eval=1)
            
            iter_n = iter_n+ bst.best_iteration
            
            yprob = bst.predict(xg_test)[:,1]
            
            dataset_blend_train[testing,nb] = yprob

        iter_label[lb] = int(float(iter_n)/num_cv)
		
# calculate precision

np.save('../data/Model1_Full.npy',dataset_blend_train)

dataset_blend_train2 = np_thresh(dataset_blend_train,0.48)

fsc = np.zeros(dataset_blend_train2.shape[0])

for nb,lb in enumerate(labels):
    train_cl = pd.read_csv('../data/train_labels_cl.csv')

    train_cl = dict(np.array(train_cl[['business_id',lb]]))

    biz_features[lb] = biz_features['business_id'].apply(lambda x: train_cl[x])

truth = np.array(biz_features[labels])

for i in range(dataset_blend_train2.shape[0]):
    
    prec,recall,fsc[i] = fscore(truth[i,:],dataset_blend_train2[i,:])

print('Fscore on this model:',np.mean(fsc))

print('time taken for cross validation',time.time()-start_time)


'''
Part3: Training and Generating submissions on the Test Set
Now that we have tuned the model parameters using cross validation, we will go ahead
and use these parameters to build 9 binary classification models
Set the submit variable to 1 only if you need to run this part, since in most cases we will only be
playing with the cross validation part
Also, we intend to use this model in an ensemble, we will store the predictions on the test
set, and use them later as features for the ensemble model
'''

if(submit):
    param['subsample'] = param['subsample']*0.8
    train_to_biz = pd.read_csv('../data/train_id.csv')

    train_image_features = np.load('../data/train_inception_7.npy',mmap_mode='r')
    
    uni_bus = train_to_biz['business_id'].unique()
    
    coll_arr = np.zeros([len(uni_bus),2048])
    
    for nb,ub in enumerate(uni_bus):
        if(nb%1000==0):
            print(nb)
        tbz = np.array(train_to_biz['business_id']==ub,dtype=bool)
        x1 = np.array(train_image_features[tbz,:])
        x1 = np.mean(x1,axis=0)
        x1 = x1.reshape([1,2048])
        coll_arr[nb,:] = x1
        
    biz_features = pd.DataFrame(uni_bus,columns=['business_id'])
    
    coll_arr = pd.DataFrame(coll_arr)
    
    frames = [biz_features,coll_arr]
    
    biz_features = pd.concat(frames,axis=1)
    
    del train_to_biz,train_image_features,coll_arr,frames
    
    model_dict = {}
    
    for nb,lb in enumerate(labels):
        train_cl = pd.read_csv('../data/train_labels_cl.csv')
    
        train_cl = dict(np.array(train_cl[['business_id',lb]]))
    
        biz_features['lb'] = biz_features['business_id'].apply(lambda x: train_cl[x])
            
        
        df_train_values = biz_features['lb']
        
        df_train_features = biz_features.drop(['business_id','lb'],axis=1)
        
        xg_train = xgb.DMatrix(df_train_features, label=df_train_values)

        bst = xgb.train(param, xg_train,iter_label[lb])

        model_dict[lb] = bst
        
        df_train_features = None
        
        df_test_features = None
        
        xg_train = None
    
    print (model_dict)
    # Predict on the test set
    
    test_to_biz = pd.read_csv('../data/test_id.csv')

    test_image_features = np.load('../data/test_inception_7.npy',mmap_mode='r')
     
    test_image_id = list(np.array(test_to_biz['photo_id'].unique()))
     
    uni_bus = test_to_biz['business_id'].unique()
     
    coll_arr = np.zeros([len(uni_bus),2048])
     
    for nb,ub in enumerate(uni_bus):
        if(nb%1000==0):
            print(nb)
        image_ids = test_to_biz[test_to_biz['business_id']==ub]['photo_id'].tolist()  
        image_index = [test_image_id.index(x) for x in image_ids]
        features = test_image_features[image_index]
        x1 = np.mean(features,axis=0)
        x1 = x1.reshape([1,2048])
        coll_arr[nb,:] = x1

        
    biz_features = pd.DataFrame(uni_bus,columns=['business_id'])
    
    coll_arr = pd.DataFrame(coll_arr)
    
    frames = [biz_features,coll_arr]
    
    biz_features = pd.concat(frames,axis=1)
    
    del coll_arr,frames,test_to_biz,test_image_features,test_image_id,image_ids,image_index,features
    
    result = np.zeros([biz_features.shape[0],9])
    
    for nb,lb in enumerate(labels):
        
        print('predicting',lb)
        df_test_features = biz_features.drop(['business_id'],axis=1)
        
        bst = model_dict[lb]
        
        yprob = bst.predict(xgb.DMatrix(df_test_features))[:,1]
        
        result[:,nb] = yprob
    
    np.save('../data/Model1_Full_result.npy',result)
    
    result = np_thresh(result,0.480)
    
    bid = np.array(biz_features['business_id'])
    
    fin = {}
    
    for i in range(result.shape[0]):
        x = result[i,:]
        li = [((q)) for q in range(9) if x[q]==1]
        fin[bid[i]] = li
        
    for j in fin.keys():
        fin[j] = ' '.join(str(e) for e in fin[j])
    
    x1 = pd.DataFrame(biz_features['business_id'])
    
    x1['labels'] = x1['business_id'].apply(lambda x: fin[x] if x in fin.keys() else '0')
    
    x1.to_csv('../data/result1.csv',index=0)
    print ("Data Saved")
