#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 01:18:59 2019

@author: felipe
"""

import json,codecs
import numpy as np
def saveHist(path,history):

    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float64:
               new_hist[key] = list(map(float, history.history[key]))

    #print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4) 

def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n

def setup_multi_gpus():
    """
    Setup multi GPU usage

    Example usage:
    model = Sequential()
    ...
    multi_model = multi_gpu_model(model, gpus=num_gpu)
    multi_model.fit()

    About memory usage:
    https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    """
    import tensorflow as tf
    from keras.utils import multi_gpu_model
    from tensorflow.python.client import device_lib

    # IMPORTANT: Tells tf to not occupy a specific amount of memory
    from keras.backend.tensorflow_backend import set_session  
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
    sess = tf.Session(config=config)  
    set_session(sess)  # set this TensorFlow session as the default session for Keras.


    # getting the number of GPUs 
    def get_available_gpus():
       local_device_protos = device_lib.list_local_devices()
       return [x.name for x in local_device_protos if x.device_type    == 'GPU']

    num_gpu = len(get_available_gpus())
    print('Amount of GPUs available: %s' % num_gpu)

    return num_gpu