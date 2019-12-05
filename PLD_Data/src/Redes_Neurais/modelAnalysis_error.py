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
MODELS_FOLDER = ROOT_FOLDER + '/modelos_pld/t_1/ModelosErro_'
#MODELS_FOLDER = ROOT_FOLDER + '/Modelos_adadelta_100epocas/'
#CLEAR_SESSION = False

N_TEST_ROWS = 3

#load trend and residue
pldTrend = pd.read_csv('pld_trend.csv', index_col=0, header=None)
pldTrend.columns = ['values']
pldSeasonal = pd.read_csv('pld_seasonal.csv', index_col=0, header=None)
pldSeasonal.columns = ['values']

N_TEST_ROWS = 3
STEPS_FORECAST = 1
# import some data to play with
X_old = pd.read_csv(INPUT_FOLDER + '/inputOld.csv').iloc[:, 1:]
X_current = pd.read_csv(INPUT_FOLDER + '/input.csv').iloc[:, 1:]
X = pd.concat([X_old, X_current], axis=1)
y_original = pd.read_csv(INPUT_FOLDER + '/output.csv').iloc[:, 1:]
y = pd.read_csv(INPUT_FOLDER + '/yError.csv').iloc[:, 1:]
reco = pd.read_csv(INPUT_FOLDER + '/reco.csv').iloc[:, 1:]

y_no_residual = y_original - y

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

STEPS_FORECAST = 1
X_train = X_norm.iloc[:-(N_TEST_ROWS + STEPS_FORECAST),:]
X_test = X_norm.iloc[-(N_TEST_ROWS + STEPS_FORECAST):-STEPS_FORECAST,:]
y_train = y_norm.iloc[:-N_TEST_ROWS, :].reset_index(drop=True)
y_test = y_norm.iloc[-N_TEST_ROWS:, :]

# train a simple classifier
n_folds = 8
n_inits = 3
MIN_NEURONS = 90     #best = 37
MAX_NEURONS = 90

#partition by folder
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=0)
CVO = kf.split(X_train, y_train)
CVO = list(CVO)

mse_matrix = pd.DataFrame(columns=['fold','n_neurons', 
                                   'mse_val_norm', 'mse_val',
                                   'mse_test_norm','mse_test',
                                   'val_rmse', 'val_std',
                                   'a', 'b'])
    
mse_comp = pd.DataFrame(columns=['mse'])
mse_comp.index.name = '#Neurons'

line = 0
hasModel = False
OPTIMIZER = 'adadelta'
ACTIVATION_HIDDEN_LAYER = 'relu'
line = 0
PLOT = False
SHOW_HISTORY = True
READ_BEST_MODEL = True
results = pd.DataFrame(columns=['rmse', 'std', 'a', 'b'], index=np.arange(MAX_NEURONS) + 1)
#resultsFolds = pd.DataFrame(columns=['rmse', 'std', 'a(mean)', 'b(mean)'], index=np.arange(MAX_NEURONS) + 1)
estimationErrors = pd.DataFrame(columns=['mean', 'std'], index=np.arange(MAX_NEURONS) + 1)

for n_neurons in range(MIN_NEURONS, MAX_NEURONS + 1):
    subplotidx = 1
    modelPath = MODELS_FOLDER + str(n_folds) + "/" + str(n_neurons) + 'neurons_' + \
            ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(7) \
            +'fold_' + OPTIMIZER
    model = load_model(modelPath + 'complete_best_weights.h5')
    
    results.plot(title="RMSE absoluto no conjunto de teste por número de neurônios na camada intermediária")
    plt.savefig('rmse_test_set.jpg')
    plt.close('all')
                                    
    plt.title('Scatter Série prevista X residual para o conjunto completo')
    y_comp = model.predict(X_norm)
    
    del model
    y_pred = y_scaler.inverse_transform(y_comp)
    estimationErrors.loc[n_neurons] = [np.mean(np.abs(y_original.iloc[STEPS_FORECAST:].price \
                         - (y_pred[STEPS_FORECAST:] + reco).value)),\
                        np.std(y_original.iloc[STEPS_FORECAST:].price \
                         - (y_pred[STEPS_FORECAST:] + reco).value)]
     
    hist = util.loadHist(modelPath + '_history.txt') 
    for fold in range(0, n_folds):                 
        if SHOW_HISTORY:
            ax1 = plt.subplot(np.ceil(n_folds/2), 2, subplotidx)
            plt.plot(np.sqrt(hist['mean_squared_error']), label='erro no treinamento')
            plt.plot(np.sqrt(hist['val_mean_squared_error']), label='erro na validação')
            plt.legend()
            ax1.set_title('Fold ' + str(subplotidx))
            plt.suptitle('RMSE na validação cruzada para  ' + str(n_neurons) + ' neurônios na camada intermediária')
            subplotidx += 1
    
    print(str(n_neurons) + " neurons")

#plot mse for validation set in folder
plt.savefig(str(n_neurons) + '_convergence_error.jpg')
plt.close('all')



y_pred = y_pred[:-STEPS_FORECAST]
y_original = y_original.iloc[STEPS_FORECAST:].reset_index(drop=True)
y_size = y_original.size
choosen_w = 3
est_amp = [94.17675855003894, 69.43507704543539, 112.68559973236269, 16.417835812860833][choosen_w]
est_freq = [0.7968374985966099, 0.8044090221275024, 0.1739844310370512, 0.23226650051018935][choosen_w]
est_mean = [-5.11368411253407, 24.509257889433268, 28.01639316680809, -6.221658471323059][choosen_w]
est_phase = [-0.2651449626303616, -0.5876011432492424, 2.7767236496885914, -0.9431117926833855][choosen_w]
t = np.arange(pldTrend.size)
senoidalCycle=est_amp*np.sin(est_freq*t+est_phase)+est_mean
y_deterministico = pldTrend.iloc[-y_size:].values + \
         pldSeasonal.iloc[-y_size:].values + \
         senoidalCycle[-y_size:].reshape(-1, 1)
n_steps = STEPS_FORECAST
plt.figure()
reco_error = reco + y_pred
plt.plot(y_original.index, reco, 'r', label='sinal com resíduo previsto')
plt.plot(y_original.index, reco_error, 'g', label='sinal com resíduo previsto e correção de erro')
plt.plot(y_original.index, y_original, 'b', label='original')
#plt.plot(y_original.index, y_deterministico, 'y', label='sinal sem resíduo')
y_std = np.std(y_original.price - reco_error.value)
plt.errorbar(y_original.index, reco_error.value, yerr=y_std, fmt='g', ecolor='black', label='desvio padrão')
plt.legend()
plt.xlabel('Amostra')
plt.ylabel('Valor PLD')
plt.title('PLD médio mensal previsto X real para o mês atual '+ str(MAX_NEURONS) + ' neurônios')
plt.show()
plt.savefig('sinal_completo_t' + str(n_steps) + '.jpg')

'''
#estimationErrors.to_csv('estimationError.csv')

plt.close('all')
y_original = pd.read_csv(INPUT_FOLDER + '/output.csv').iloc[:, 1:]
y_pred = y_scaler.inverse_transform(y_comp)
plt.figure()
plt.title('Histograma do erro entre o sinal reconstruido e previsão')
NUM_ELEMENTS = 28
x = np.arange(NUM_ELEMENTS-1)
plt.plot(x, y_original.iloc[STEPS_FORECAST:], 'b', label='sinal original')
plt.plot(x, y_pred[STEPS_FORECAST:] + reco, 'r', label='sinal previsto reconstruído')
plt.plot(x, reco, 'g', label='sinal previsto reconstruído com a adição da rede corretora de erro')
plt.legend()
plt.savefig('original_pred_rede_erro_tempo.jpg')
'''

plt.figure()
plt.title('Histograma do erro entre o sinal reconstruido e previsão')
plt.hist( (y_original.price - reco.value), color='red', alpha=0.5, label='erro entre o sinal original e sinal previsto reconstruído' )
plt.hist( (y_original.price \
                         - reco_error.value), color='green', alpha=0.5, label='erro entre o sinal original e sinal previsto reconstruído com rede para diminuir erro')
plt.legend()
plt.savefig('original_pred_rede_erro_hist.jpg')

#azul como sendo truth, vermelho reco, e verde reco mais nn
