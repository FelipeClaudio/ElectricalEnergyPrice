#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 07:30:03 2019

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

best_models = [61, 63, 69, 88 , 87, 34, 57, 85, 29, 66, 74]

ROOT_FOLDER  = '/home/felipe/Materias/TCC/PLD_Data/src/Redes_Neurais'
INPUT_FOLDER = ROOT_FOLDER + '/InputNN'
MODELS_FOLDER = ROOT_FOLDER + '/modelos_pld/t_'
plt.close('all')

OPTIMIZER = 'adadelta'
ACTIVATION_HIDDEN_LAYER = 'relu'
line = 0
PLOT = False
SHOW_HISTORY = True
READ_BEST_MODEL = True

n_steps = 0
n_folds = 8
n_inits = 3

N_TEST_ROWS = 3

y_error = [None] * len(best_models)
y_error_test = [None] * len(best_models)
y_std = [None] * len(best_models)
y_std_test = [None] * len(best_models)
model_params = pd.DataFrame(columns=['a_comp', 'b_comp', 'a_test', 'b_test'], index=np.arange(len(best_models)))
for n_neurons in best_models:
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
    N_TEST_ROWS = 3
    X_train = X_norm.iloc[:-(N_TEST_ROWS + STEPS_FORECAST),:]
    y_train = y_norm.iloc[STEPS_FORECAST:-N_TEST_ROWS, :].reset_index(drop=True)
    
    current_models_folder = MODELS_FOLDER +  str(n_steps) + '/Modelos_' + str(n_folds) + '/'
    print(str(n_neurons) + ' neurons')
    if (n_steps > 1):
        modelPath = current_models_folder + str(n_neurons) + ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(7) \
                        +'fold_' + OPTIMIZER
    else:
        modelPath = current_models_folder + str(n_neurons) + 'neurons_'+ ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(7) \
                        +'fold_' + OPTIMIZER
    K.clear_session()
    
    #choose between best model or last model
    if READ_BEST_MODEL:
        model = load_model(modelPath + 'best_weights.h5')
        
    y_comp = model.predict(X_norm)
    
    del model
    
    y_pred = y_scaler.inverse_transform(y_comp[STEPS_FORECAST:, :])
    y_original_pred = y.iloc[STEPS_FORECAST:, :].reset_index(drop=True)
    
    t = np.arange(y_pred.size)
    plt.figure()
    plt.title('Previsão da série residual desnormalizada para ' + str(n_steps) + ' passos a frente')
    plt.plot(t, y_pred, 'b', label='série residual original')
    plt.plot(t, y_original_pred, 'r', label='série residual prevista')
    plt.legend()
    plt.savefig(str(n_steps) + '_best_prev.jpg')
    
    print(str(n_neurons) + " neurons")

    y_error[n_steps] = np.mean(abs(y_pred - y_original_pred))
    y_error_test[n_steps] = np.mean(abs(y_pred[-N_TEST_ROWS:] - y_original_pred.iloc[-N_TEST_ROWS:]))
    y_std[n_steps] = np.std(y_pred - y_original_pred)
    y_std_test[n_steps] = np.mean(abs(y_pred[-N_TEST_ROWS:] - y_original_pred.iloc[-N_TEST_ROWS:]))
    params_comp = np.polyfit(y_original_pred.price, y_pred , 1)
    params_test = np.polyfit(y_original_pred.iloc[-N_TEST_ROWS:].price, y_pred[-N_TEST_ROWS:], 1)
    model_params.loc[n_steps] = [params_comp[0][0], params_comp[1][0], params_test[0][0], params_test[1][0]]
    model_params.to_csv('model_params.csv')
    
    plt.figure()
    plt.scatter(y_original_pred, y_pred)
    plt.xlabel('Série prevista')
    plt.ylabel('Série original')
    plt.title('Serie prevista x original para o dataset completo e previsão de ' + str(n_steps) +' passos a frente')
    y_fit_comp = model_params.loc[n_steps]['a_comp'] * y_original_pred + model_params.loc[n_steps]['b_comp']
    plt.plot(y_original_pred, y_fit_comp, 'r', label='a=' + str(model_params.loc[n_steps]['a_comp'])+ \
             ' b=' + str(model_params.loc[n_steps]['b_comp']) )
    plt.legend()
    plt.savefig(str(n_steps) + '_linear_fit_comp.jpg')
    
    plt.figure()
    plt.scatter(y_original_pred.iloc[-N_TEST_ROWS:] , y_pred[-N_TEST_ROWS:])
    plt.xlabel('Série prevista')
    plt.ylabel('Série original')
    plt.title('Serie prevista x original para o dataset de teste e previsão de ' + str(n_steps) +' passos a frente')
    y_fit_test = model_params.loc[n_steps]['a_test'] * y_original_pred.iloc[-N_TEST_ROWS:] + model_params.loc[n_steps]['b_test']
    plt.plot(y_original_pred.iloc[-N_TEST_ROWS:] , y_fit_test, 'r', label='a=' + str(model_params.loc[n_steps]['a_test'])+ \
             ' b=' + str(model_params.loc[n_steps]['b_test']) )
    plt.legend()
    plt.savefig(str(n_steps) + '_linear_fit_test.jpg')
    
    n_steps = n_steps + 1
    
plt.figure()
t2 = np.arange(len(y_error))
plt.title('Erro pelo número de passos de previsão para o dataset completo')
plt.xlabel('Número de passos à frente')
plt.ylabel('Erro médio')
plt.plot(t2, y_error, label='erro médio absoluto pelo número de passos')
plt.errorbar(t2, y_error, yerr=y_std, fmt='r', ecolor='black', label='desvio padrão')
plt.legend()
plt.savefig('mean_std_comp_best_prev.jpg')

plt.figure()
t2 = np.arange(len(y_error_test))
plt.title('Erro pelo número de passos de previsão para o dataset de teste')
plt.plot(t2, y_error_test, label='erro médio absoluto pelo número de passos')
plt.xlabel('Número de passos à frente')
plt.ylabel('Erro médio')
plt.errorbar(t2, y_error_test, yerr=y_std_test, fmt='r', ecolor='black', label='desvio padrão')
plt.legend()
plt.savefig('mean_std_test_best_prev.jpg')