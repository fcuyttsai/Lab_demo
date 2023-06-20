#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,callbacks, models
from functools import partial
import matplotlib.pyplot as plt

import numpy as np
sys.path.append('code')
from tfrecord import *  
from tensorflow.keras.models import load_model
datapath='database' #Data path

os.listdir(datapath) #print out the files in the target folder

model_ = load_model('CNN_model.h5')
train_dataset = get_dataset(datapath+'/train_data.tfrecord',batchsize=1)


# In[2]:


train_iterator= iter(train_dataset)
count =0
for sample in train_iterator:
    test_inputs=sample[0]
    test_outputs=sample[1]
    predictions = model_.predict(test_inputs)
    predicted_labels = np.argmax(predictions, axis=1)
    print('time index:', count,' Predicted:',predicted_labels)
    count=count+1
    time.sleep(2)


# In[ ]:




