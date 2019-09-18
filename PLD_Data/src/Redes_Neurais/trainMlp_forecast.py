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
from keras.optimizers import Adadelta
import keras.callbacks as callbacks
import warnings
from pylab import rcParams
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras import backend as K
import copy
import util_neural_network as util
from keras.callbacks import ModelCheckpoint
import time

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
MODELS_FOLDER = ROOT_FOLDER + '/modelos_pld/t_'
#CLEAR_SESSION = False

N_TEST_ROWS = 3
min_steps = 0
max_steps = 1

for n_steps in range(min_steps, max_steps + 1):
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
    
    STEPS_FORECAST = n_steps
    X_train = X_norm.iloc[:-(N_TEST_ROWS + STEPS_FORECAST),:]
    y_train = y_norm.iloc[STEPS_FORECAST:-N_TEST_ROWS, :].reset_index(drop=True)
    
    # train a simple classifier
    #n_folds = 3
    n_inits = 3
    MIN_NEURONS = 1 #refazer o 40, 73, 74, 78
    MAX_NEURONS = 90    
    
    classifiers = {}
    mse_matrix = pd.DataFrame(columns=['fold','n_neurons', 'mse_train' , 'mse_test'])
    line = 0
    OPTIMIZER = 'adadelta'
    ACTIVATION_HIDDEN_LAYER = 'relu'
    MULTI_GPU = False
    
    min_folder = 8
    max_folder = 8
    for n_folds in range (min_folder, max_folder + 1):
        kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=0)
        CVO = kf.split(X_train, y_train)
        CVO = list(CVO)
        
        t = time.time()
        for n_neurons in range(MIN_NEURONS, MAX_NEURONS + 1):
            for ifold in range(n_folds):
                best_model = None
                train_id, test_id = CVO[ifold]
               
                best_mse = 9999999
                current_models_folder = MODELS_FOLDER +  str(n_steps) + '/Modelos_' + str(n_folds) + '/'
                print(str(n_neurons) + ' neurons')
                path = current_models_folder + str(n_neurons) + ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(ifold) \
                                    +'fold_' + OPTIMIZER
                print (path)
                                    
                #random initialization for neural networks
                for i_init in range(n_inits):
                    print ('Processing: Fold %i of %i --- Init %i of %i'%(
                            ifold+1, n_folds, 
                            i_init+1, n_inits))
                    model = Sequential()
                    model.add(Dense(n_neurons,
                                    input_dim=X.shape[1],
                                    init='he_uniform', 
                                    activation=ACTIVATION_HIDDEN_LAYER))
                    model.add(Dense(1, 
                                    kernel_initializer='uniform', 
                                    activation='linear')) 
                    
                    #choose optimizer
                    if OPTIMIZER == 'sgd':
                        opt = SGD(lr=0.00001, decay=1e-3, momentum=0.9)
                    elif OPTIMIZER == 'adadelta':
                        opt = Adadelta(lr=0.1)
                    else:
                        opt = OPTIMIZER
                    
                    model.compile(loss='mean_squared_error',
                                  optimizer=opt,
                                  metrics=['mse'])
                    
                    #use mode compiled for gpu optimization
                    if MULTI_GPU:
                        from keras.utils import multi_gpu_model
                        num_gpus = util.setup_multi_gpus()
                        model = multi_gpu_model(model, gpus=num_gpus)
                    
                    #set callbacks
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
                                              nb_epoch=10000, 
                                              batch_size=4, 
                                              callbacks=[earlyStopping, checkpoint], 
                                              #callbacks=[checkpoint], 
                                              verbose=0, 
                                              validation_data=(X_train.iloc[test_id, :],
                                                               y_train.iloc[test_id, :]), 
                                              shuffle=True)
                    
                    #get best model
                    if np.min(hist.history['val_mean_squared_error']) < best_mse:
                        best_loss = np.min(hist.history['val_mean_squared_error'])
                        best_model = copy.deepcopy(model)
        
                        util.saveHist(path +'_history.txt', hist)
                   
                    del model
                    del hist
                    best_model.save(path + '.h5')
                    del best_model
                    
                    
            model = Sequential()
            model.add(Dense(n_neurons,
                            input_dim=X.shape[1],
                            init='he_uniform', 
                            activation=ACTIVATION_HIDDEN_LAYER))
            model.add(Dense(1, 
                            kernel_initializer='uniform', 
                            activation='linear'))
            model.compile(loss='mean_squared_error',
                          optimizer=opt,
                          metrics=['mse'])
            checkpoint = ModelCheckpoint(path + 'complete_best_weights.h5', 
                                 monitor='mean_squared_error', 
                                 save_best_only=True, 
                                 mode='min')
            hist = model.fit(X_train, 
                              y_train,
                              nb_epoch=100, 
                              batch_size=4, 
                              callbacks=[earlyStopping, checkpoint], 
                              #callbacks=[checkpoint], 
                              verbose=0)
            util.saveHist(path +'_historyComplete.txt', hist)
            K.clear_session()
            del model
            del hist
        
        elapsed = time.time() - t
        print(elapsed)
