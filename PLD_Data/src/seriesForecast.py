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

ROOT_FOLDER  = '/home/felipe/Materias/TCC/PLD_Data/src'
INPUT_FOLDER = ROOT_FOLDER + '/InputNN'
CLEAR_SESSION = True

N_VALIDATION_ROWS = 3

# import some data to play with
X_old = pd.read_csv(INPUT_FOLDER + '/inputOld.csv').iloc[:, 1:]
X_current = pd.read_csv(INPUT_FOLDER + '/input.csv').iloc[:, 1:]
X = pd.concat([X_old, X_current], axis=1)
#y = pd.read_csv(INPUT_FOLDER + '/output.csv').iloc[:, 1:]
y = pd.read_csv(INPUT_FOLDER + '/output_residual.csv').iloc[:, 1:]
X_train = X.iloc[:-N_VALIDATION_ROWS,:]
X_validation = X.iloc[-N_VALIDATION_ROWS:,:]
y_train = y.iloc[:-N_VALIDATION_ROWS, :]
y_validation = y.iloc[-N_VALIDATION_ROWS:, :]

#Data scaling
X_train_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_norm = X_train_scaler.transform(X_train)

Y_train_scaler = preprocessing.StandardScaler().fit(y_train)
y_train_norm = Y_train_scaler.transform(y_train)

X_val_norm = X_train_scaler.transform(X_validation)

# train a simple classifier
n_folds = 3
n_inits = 3
MIN_NEURONS = 30 #best = 37
MAX_NEURONS = 40
norm = 'mapstd'

kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=0)
CVO = kf.split(X_train, y_train)
CVO = list(CVO)

classifiers = {}
mse_matrix = pd.DataFrame(columns=['fold','n_neurons', 'mse_train' , 'mse_val'])
line = 0
hasModel = False
for ifold in range(n_folds):
    train_id, test_id = CVO[ifold]
    
    # normalize data based in train set
    if norm == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(X_train.iloc[train_id,:])
    elif norm == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(X_train.iloc[train_id,:])
    elif norm == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(X_train.iloc[train_id,:])
        
    norm_data = scaler.transform(X_train)
   
    
    for n_neurons in range(MIN_NEURONS, MAX_NEURONS + 1):
        best_init = 0
        best_loss = 9999999
        print(str(n_neurons) + ' neurons')
        for i_init in range(n_inits):
            print ('Processing: Fold %i of %i --- Init %i of %i'%(
                    ifold+1, n_folds, 
                    i_init+1, n_inits))
            model = Sequential()
            model.add(Dense(X_train.shape[1],
                            input_dim=X.shape[1],
                            init='identity',))
                            #trainable=False))
            model.add(Dense(n_neurons,
                            init='uniform', 
                            activation='tanh'))
            model.add(Dense(1, 
                            init='uniform', 
                            activation='linear')) 
            
            #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.7)
            model.compile(loss='mean_squared_error',
                          optimizer='adam',
                          metrics=['accuracy'])
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=25, 
                                                    verbose=0, 
                                                    mode='auto')
            # Train model
            init_trn_desc = model.fit(X_train_norm[train_id, :], 
                                      y_train_norm[train_id, :],
                                      nb_epoch=100000000, 
                                      batch_size=128, 
                                      callbacks=[earlyStopping], 
                                      verbose=0, 
                                      validation_data=(X_train_norm[test_id, :],
                                                       y_train_norm[test_id, :]), 
                                      shuffle=True)
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                classifiers[line] = model
                hasModel = True
                print('aqui')
        if hasModel:
            Y_scaler = preprocessing.StandardScaler().fit(y)
            y_pred_train = Y_train_scaler.inverse_transform(classifiers[line].predict(X_train_norm[test_id]))
            y_pred_val = Y_train_scaler.inverse_transform(classifiers[line].predict(X_val_norm))
            mse_matrix.loc[line] = [ifold, n_neurons,
                          mean_squared_error(y_train_norm[test_id], y_pred_train),
                          mean_squared_error(y_validation, y_pred_val)]
        else:
            mse_matrix.loc[line] = [ifold, n_neurons, np.Infinity, np.Infinity]
        line += 1
        hasModel = False
        #model.summary()
        if CLEAR_SESSION:
            K.clear_session()

if not CLEAR_SESSION:
    CHOOSEN_CLASSIFIER = 0
    
    #X_scaler = preprocessing.StandardScaler().fit(X_validation)
    X_scaler = preprocessing.StandardScaler().fit(X)
    y_scaler = preprocessing.StandardScaler().fit(y)
    X_norm = X_scaler.transform(X)
    model_validation = classifiers[CHOOSEN_CLASSIFIER]
    y_pred = y_scaler.inverse_transform(model_validation.predict(X_norm))
    
    #plt.plot(y_validation.reset_index(drop=True), label='y_val')
    plt.plot(y, label='y_val')
    plt.plot(y_pred, label='y_pred')
    plt.legend()
    plt.show()
    
    
    plt.figure()
    X_norm = X_scaler.transform(X_validation)
    y_pred = y_scaler.inverse_transform(model_validation.predict(X_val_norm))
    plt.plot(y_validation.reset_index(drop=True), label='y_val')
    plt.plot(y_pred, label='y_pred')
    plt.legend()
    plt.show()

mse_matrix = mse_matrix.groupby(['n_neurons']).sum().drop(columns= ['fold'])
mse_matrix = mse_matrix.replace([np.inf, -np.inf], 0)

#plt.figure()
#plt.subplot(2, 1, 1)
#plt.stem(mse_matrix.index, mse_matrix.iloc[:,0], label='mse_train')
#plt.legend()
#plt.subplot(2, 1, 2)
#plt.stem(mse_matrix.index, mse_matrix.iloc[:,1], label='mse_validation')
mse_matrix.plot(title='MSE na validação cruzada e no dataset de validação para o resíduo do PLD usando tanh(x) na camada escondida')
plt.xlabel('Número Neurônios')
plt.ylabel('MSE')
plt.legend()
plt.show()
