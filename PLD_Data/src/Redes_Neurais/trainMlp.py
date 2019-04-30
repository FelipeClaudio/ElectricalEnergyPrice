#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:10:19 2019

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:23:34 2019

@author: felipe
"""
import os
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
import warnings
from pylab import rcParams
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras import backend as K
import copy
import util_neural_network as util
from keras.callbacks import ModelCheckpoint

plt.close('all')

##Settings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

ROOT_FOLDER  = '/home/felipe/Materias/TCC/PLD_Data/src/Redes_Neurais'
INPUT_FOLDER = ROOT_FOLDER + '/InputNN'
MODELS_FOLDER = ROOT_FOLDER + '/Modelos/'
#CLEAR_SESSION = False

N_TEST_ROWS = 3

# import some data to play with
X_old = pd.read_csv(INPUT_FOLDER + '/inputOld.csv').iloc[:, 1:]
X_current = pd.read_csv(INPUT_FOLDER + '/input.csv').iloc[:, 1:]
X = pd.concat([X_old, X_current], axis=1)
#y = pd.read_csv(INPUT_FOLDER + '/output.csv').iloc[:, 1:]
y = pd.read_csv(INPUT_FOLDER + '/output_residual.csv').iloc[:, 1:]

norm = 'mapminmax'
if norm == 'mapstd':
    scaler = preprocessing.StandardScaler()
elif norm == 'mapstd_rob':
    scaler = preprocessing.RobustScaler()
elif norm == 'mapminmax':
    scaler = preprocessing.MinMaxScaler()

#Data scaling
X_scaler = copy.deepcopy(scaler).fit(X)
X_norm = pd.DataFrame(data=X_scaler.transform(X), columns=X.columns, index=X.index)

y_scaler = copy.deepcopy(scaler).fit(y)
y_norm = pd.DataFrame(data=y_scaler.transform(y), columns=y.columns, index=y.index)

X_train = X_norm.iloc[:-N_TEST_ROWS,:]
X_test = X_norm.iloc[-N_TEST_ROWS:,:]
y_train = y_norm.iloc[:-N_TEST_ROWS, :]
y_test = y_norm.iloc[-N_TEST_ROWS:, :]

# train a simple classifier
n_folds = 8
n_inits = 3
MIN_NEURONS = 1 #best = 37
MAX_NEURONS = 120

kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=0)
CVO = kf.split(X_train, y_train)
CVO = list(CVO)

classifiers = {}
mse_matrix = pd.DataFrame(columns=['fold','n_neurons', 'mse_train' , 'mse_test'])
line = 0
OPTIMIZER = 'sgd'
ACTIVATION_HIDDEN_LAYER = 'relu'
USE_GPU = False

for n_neurons in range(MIN_NEURONS, MAX_NEURONS + 1):
    for ifold in range(n_folds):
        best_model = None
        train_id, test_id = CVO[ifold]
       
        best_init = 0
        best_mse = 9999999
        print(str(n_neurons) + ' neurons')
        path = MODELS_FOLDER + str(n_neurons) + 'neurons_' + \
               ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(ifold) \
                            +'fold_' + OPTIMIZER
        for i_init in range(n_inits):
            print ('Processing: Fold %i of %i --- Init %i of %i'%(
                    ifold+1, n_folds, 
                    i_init+1, n_inits))
            model = Sequential()
            model.add(Dense(n_neurons,
                            input_dim=X.shape[1],
                            init='uniform', 
                            activation=ACTIVATION_HIDDEN_LAYER))
                            #trainable=False))
            #model.add(Dense(n_neurons,
            #                init='uniform', 
            #                activation=ACTIVATION_HIDDEN_LAYER))
            model.add(Dense(1, 
                            kernel_initializer='he_uniform', 
                            activation='linear')) 
            
            sgd = SGD(lr=0.001, decay=1e-3, momentum=0.3)
            model.compile(loss='mean_squared_error',
                          optimizer=sgd,
                          metrics=['mse'])
            
            if USE_GPU:
                from keras.utils import multi_gpu_model
                num_gpus = util.setup_multi_gpus()
                model = multi_gpu_model(model, gpus=num_gpus)
            
            earlyStopping = callbacks.EarlyStopping(monitor='val_mean_squared_error', 
                                                    patience=25, 
                                                    verbose=0, 
                                                    mode='auto')
            checkpoint = ModelCheckpoint(path + 'best_weights.h5', 
                                         monitor='val_mean_squared_error', 
                                         save_best_only=True, 
                                         mode='min')
            # Train model
            hist = model.fit(X_train.iloc[train_id, :], 
                                      y_train.iloc[train_id, :],
                                      nb_epoch=1000, 
                                      batch_size=128, 
                                      callbacks=[earlyStopping, checkpoint], 
                                      verbose=0, 
                                      validation_data=(X_train.iloc[test_id, :],
                                                       y_train.iloc[test_id, :]), 
                                      shuffle=True)
            if np.min(hist.history['val_mean_squared_error']) < best_mse:
                best_init = i_init
                best_loss = np.min(hist.history['val_mean_squared_error'])
                best_model = copy.deepcopy(model)

                util.saveHist(path +'_history.txt', hist)
                del model
                del hist
                print('aqui')
           
            best_model.save(path + '.h5')
            del best_model
            K.clear_session()
