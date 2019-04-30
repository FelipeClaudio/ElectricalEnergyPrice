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
import copy
from keras.models import load_model
import util_neural_network as util


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
MIN_NEURONS = 120 #best = 37
MAX_NEURONS = 120
norm = 'mapstd'

kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=0)
CVO = kf.split(X_train, y_train)
CVO = list(CVO)

mse_matrix = pd.DataFrame(columns=['fold','n_neurons', 'mse_val'])
line = 0
hasModel = False
OPTIMIZER = 'sgd'
ACTIVATION_HIDDEN_LAYER = 'relu'
line = 0
PLOT = False
SHOW_HISTORY = True
READ_BEST_MODEL = True

for n_neurons in range(MIN_NEURONS, MAX_NEURONS + 1):
    if PLOT:
        plt.figure()
        
    for ifold in range(n_folds):
        train_id, test_id = CVO[ifold]
        modelPath = MODELS_FOLDER + str(n_neurons) + 'neurons_' + \
                    ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(ifold) \
                    +'fold_' + OPTIMIZER
                    
        K.clear_session()       
        if READ_BEST_MODEL:
            model = load_model(modelPath + 'best_weights.h5')
        else:
            model = load_model(modelPath + '.h5')

        '''
        y_pred_train = y_scaler.inverse_transform(model.predict(X_train.iloc[test_id]))
        y_pred_test = y_scaler.inverse_transform(model.predict(X_test))
        mse_matrix.loc[line] = [ifold, n_neurons,
                      mean_squared_error(y.iloc[test_id], y_pred_train),
                      mean_squared_error(y_test, y_pred_test)]
        '''
        
        hist = util.loadHist(modelPath + '_history.txt')
        min_mse = np.min(hist['val_mean_squared_error'])
        mse_matrix.loc[line] = [ifold, n_neurons, min_mse]
        
        line += 1
        del model       
        if SHOW_HISTORY:
            plt.subplot(np.ceil(n_folds/2), 2, line)
            plt.plot(hist['val_mean_squared_error'])         

    if PLOT:
        plt.show()
        
    print (str(n_neurons) + ' neurons')
  

plt.figure()
plt.title('MSE médio no conjunto de validação por número de neurônios na camada intermediária')    
mse_mean = mse_matrix.groupby(['n_neurons']).mean().drop(columns= ['fold'])
plt.plot(mse_mean.iloc[:, -1])

'''        
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
    X_norm = X_scaler.transform(X_test)
    y_pred = y_scaler.inverse_transform(model_validation.predict(X_test_norm))
    plt.plot(y_test.reset_index(drop=True), label='y_val')
    plt.plot(y_pred, label='y_pred')
    plt.legend()
    plt.show()

mse_matrix = mse_matrix.groupby(['n_neurons']).mean().drop(columns= ['fold'])
mse_matrix = mse_matrix.replace([np.inf, -np.inf], 0)

#plt.figure()
#plt.subplot(2, 1, 1)
#plt.stem(mse_matrix.index, mse_matrix.iloc[:,0], label='mse_train')
#plt.legend()
#plt.subplot(2, 1, 2)
#plt.stem(mse_matrix.index, mse_matrix.iloc[:,1], label='mse_validation')
mse_matrix.plot(title='MSE na validação cruzada e no dataset de teste para o resíduo do PLD usando relu(x) na camada escondida')
plt.xlabel('Número Neurônios')
plt.ylabel('MSE')
plt.legend()
plt.show()
'''