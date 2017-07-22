# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:23:46 2017

@author: athir
"""

'''
Using a pretrained net to generate features for images
Assuming the script is run on a GPU, it takes close to 6 hours to complete (This can be optimized by changing batch size)
The output files are close to 4GB each for train and test
The inception net weights and architecture need to be downloaded from here: https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md
'''


import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
from mxnet import model
import pandas as pd
import time
import datetime

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

bs = 1

def PreprocessImage(path, show_img=False,invert_img=False):
    img = io.imread(path)
    if(invert_img):
        img = np.fliplr(img)
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (299, 299))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (299, 299, 3) to (3, 299, 299)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean 
    normed_img = sample - 128
    normed_img /= 128.
    return np.reshape(normed_img,(1,3,299,299))

start_time =  datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 

prefix = "model/inception-7/Inception-7"

num_round = 1

network = model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=bs)

inner = network.symbol.get_internals()

inner_feature = inner['flatten_output']

fea_ext = model.FeedForward(ctx=mx.cpu(),symbol=inner_feature,numpy_batch_size=bs,arg_params=network.arg_params,aux_params=network.aux_params,allow_extra_params=True)


# biz_ph = pd.read_csv('../data/train_id.csv')

# ph = biz_ph['photo_id'].unique().tolist()

# train_label = pd.read_csv('../data/train_labels_cl.csv')

# train_ph = train_label['business_id'].unique().tolist()
 
# feat_holder = np.zeros([len(ph),2048])

# for num_ph,photo in enumerate(biz_ph):
    # print (photo)
    # fp = '../data/train/'+str(photo)+'.jpg'
    # try:
        # feat_holder[num_ph,:]=fea_ext.predict(PreprocessImage(fp,show_img=False,invert_img=False))
        # print ("converted", fp, num_ph)
    # except FileNotFoundError:
        # pass
# print(feat_holder.shape)

# np.save('../data/train_processed.npy',feat_holder)

# end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 

# print("start ", start_time, "end ", end_time)



biz_ph = pd.read_csv('../data/test_id.csv')

ph = biz_ph['photo_id'].unique().tolist()

test_label = pd.read_csv('../data/test_labels_cl.csv')

test_ph = test_label['business_id'].unique().tolist()
 
feat_holder = np.zeros([len(ph),2048])


for num_ph,photo in enumerate(ph):
    print (photo)
    fp = '../data/test/'+str(photo)+'.jpg'
    try:
        feat_holder[num_ph,:]=fea_ext.predict(PreprocessImage(fp,show_img=False,invert_img=False))
        print ("converted", fp, num_ph)
    except FileNotFoundError:
        pass
print(feat_holder.shape)

np.save('../data/test_processed.npy',feat_holder)

end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 

print("start ", start_time, "end ", end_time)

