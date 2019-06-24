#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 02:22:24 2019

@author: felipe
"""

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
MODELS_FOLDER = ROOT_FOLDER + '/Modelos_'
#CLEAR_SESSION = False

N_TEST_ROWS = 4

# import some data to play with
X_old = pd.read_csv(INPUT_FOLDER + '/inputOld.csv').iloc[:, 1:]
X_current = pd.read_csv(INPUT_FOLDER + '/input.csv').iloc[:, 1:]
X = pd.concat([X_old, X_current], axis=1)
y_original = pd.read_csv(INPUT_FOLDER + '/output.csv').iloc[:, 1:]
y = pd.read_csv(INPUT_FOLDER + '/output_residual.csv').iloc[:, 1:]
#y.append({'price':192.10}, ignore_index=True)

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

SHIFT_RESULT = 1

X_train = X_norm.iloc[:-N_TEST_ROWS,:]
X_test = X_norm.iloc[-N_TEST_ROWS:,:]
y_train = y_norm.iloc[SHIFT_RESULT:-N_TEST_ROWS + SHIFT_RESULT, :].reset_index(drop=True)
y_test = y_norm.iloc[-N_TEST_ROWS + SHIFT_RESULT:, :]

y_original = y_original.iloc[1:, :].reset_index(drop=True)

# train a simple classifier
#n_folds = 3
n_inits = 3
N_NEURONS = 52

classifiers = {}
mse_matrix = pd.DataFrame(columns=['fold','n_neurons', 'mse_train' , 'mse_test'])
line = 0
OPTIMIZER = 'adadelta'
ACTIVATION_HIDDEN_LAYER = 'relu'
MULTI_GPU = False

min_folder = 3
max_folder = 8
path = ROOT_FOLDER

#load trend and residue
pldTrend = pd.read_csv('pld_trend.csv', index_col=0, header=None)
pldTrend.columns = ['values']
pldSeasonal = pd.read_csv('pld_seasonal.csv', index_col=0, header=None)
pldSeasonal.columns = ['values']

#param for normalizes signal composed by residual + senoidal cycle
est_amp = 69.43507704543539
est_freq = 0.8044090221275024
est_mean = 24.509257889433268
est_phase = -0.5876011432492424

t = np.arange(pldTrend.size)
senoidalCycle=est_amp*np.sin(est_freq*t+est_phase)+est_mean



t = time.time()
            
#choose optimizer
if OPTIMIZER == 'sgd':
    opt = SGD(lr=0.001, decay=1e-3, momentum=0.9)
elif OPTIMIZER == 'adadelta':
    opt = Adadelta()
else:
    opt = OPTIMIZER
                         
#set callbacks
earlyStopping = callbacks.EarlyStopping(monitor='val_mean_squared_error', 
                                        patience=25, 
                                        verbose=0, 
                                        mode='auto')
checkpoint = ModelCheckpoint(path + 'best_weights_t1.h5', 
                             monitor='val_mean_squared_error', 
                             save_best_only=True, 
                             mode='min')               
            
model = Sequential()
model.add(Dense(N_NEURONS,
                input_dim=X.shape[1],
                init='he_uniform', 
                activation=ACTIVATION_HIDDEN_LAYER))
model.add(Dense(1, 
                kernel_initializer='uniform', 
                activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['mse'])
checkpoint = ModelCheckpoint(path + 'complete_best_weights_t1.h5', 
                     monitor='mean_squared_error', 
                     save_best_only=True, 
                     mode='min')
hist = model.fit(X_train, 
                  y_train,
                  nb_epoch=100, 
                  batch_size=32, 
                  callbacks=[earlyStopping, checkpoint], 
                  #callbacks=[checkpoint], 
                  verbose=0)
#util.saveHist(path +'_historyComplete.txt', hist)

elapsed = time.time() - t
print(elapsed)


y_comp = model.predict(X_norm.iloc[:-SHIFT_RESULT]) 
reco = y_scaler.inverse_transform(y_comp) + pldTrend.iloc[-28+SHIFT_RESULT:].values + pldSeasonal.iloc[-28+SHIFT_RESULT:].values + \
senoidalCycle[-28+SHIFT_RESULT:].reshape(-1, 1)
#plt.plot(t, senoidalCycle)


'''
reco2 = pldTrend.iloc[-28:].values + pldSeasonal.iloc[-28:].values + \
senoidalCycle[-28:].reshape(-1, 1)

y2 = y_original - y
t2 = np.arange(y2.size)

plt.figure()
plt.plot(t2, reco2, 'r')
plt.plot(t2, y2, 'b')
plt.show()
'''

plt.figure()
plt.scatter(reco, y_original)
plt.title('Reco X Original')
plt.xlabel('Reco')
plt.ylabel('Original')
params = np.polyfit(reco.reshape(-1), y_original, 1)
y_fit = params[0] * reco + params[1]
plt.plot(reco, y_fit, label='a=' + str(params[0])+ ' b=' + str(params[1]))
plt.legend()
plt.show()

np.max(np.abs(y_original - reco))
np.std(y_original - reco)
np.min(np.abs(y_original - reco))

plt.figure()
plt.plot(y_original.index, reco, 'r', label='reco')
plt.plot(y_original.index, y_original, 'b', label='original')
plt.legend()
plt.xlabel('Amostra')
plt.ylabel('Valor PLD')
plt.title('PLD médio mensal Previsto X Real para o mês seguinte 52 neurônios')
plt.show() 
