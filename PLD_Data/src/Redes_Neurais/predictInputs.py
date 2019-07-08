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
from keras.models import load_model
import sys
sys.path.append("..")
import locale
locale.setlocale(locale.LC_TIME, "en_US.UTF-8") 
import utilities

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
ONS_DIR = '/home/felipe/Materias/TCC/PLD_Data/ONS_DATA_COMPLETE'
FINAL_DATE = '12/2018'
INITIAL_DATE = '01/2002'
#CLEAR_SESSION = False

TEST_ROWS = 12

# import some data to play with
X = pd.read_csv(INPUT_FOLDER+'/ENA_tsa_decomposition.csv')['residual']
X_aux = pd.read_csv(INPUT_FOLDER+'/ENA_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]
X = pd.concat([X, pd.read_csv(INPUT_FOLDER+'/Useful Afluent Flow Sum_tsa_decomposition.csv')['residual']], axis=1)
X_aux = pd.concat([X_aux, pd.read_csv(INPUT_FOLDER+'/Useful Afluent Flow Sum_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]], axis=1)
X = pd.concat([X, pd.read_csv(INPUT_FOLDER+'/Afluent Flow Sum_tsa_decomposition.csv')['residual']], axis=1)
X_aux = pd.concat([X_aux, pd.read_csv(INPUT_FOLDER+'/Afluent Flow Sum_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]], axis=1)
X = pd.concat([X, pd.read_csv(INPUT_FOLDER+'/Load Energy_tsa_decomposition.csv')['residual']], axis=1).iloc[-X.index.size:,:]
X_aux = pd.concat([X_aux, pd.read_csv(INPUT_FOLDER+'/Load Energy_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]], axis=1)
X = pd.concat([X, pd.read_csv(INPUT_FOLDER+'/UTE Generated Energy_tsa_decomposition.csv')['residual']], axis=1)
X_aux = pd.concat([X_aux, pd.read_csv(INPUT_FOLDER+'/UTE Generated Energy_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]], axis=1)
X = pd.concat([X, pd.read_csv(INPUT_FOLDER+'/UN Generated Energy_tsa_decomposition.csv')['residual']], axis=1)
X_aux = pd.concat([X_aux, pd.read_csv(INPUT_FOLDER+'/UN Generated Energy_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]], axis=1)
X = pd.concat([X, pd.read_csv(INPUT_FOLDER+'/UHE Generated Energy_tsa_decomposition.csv')['residual']], axis=1)
X_aux = pd.concat([X_aux, pd.read_csv(INPUT_FOLDER+'/UHE Generated Energy_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]], axis=1)
X = pd.concat([X, pd.read_csv(INPUT_FOLDER+'/Total Stored Energy_tsa_decomposition.csv')['residual']], axis=1)
X_aux = pd.concat([X_aux, pd.read_csv(INPUT_FOLDER+'/Total Stored Energy_tsa_decomposition.csv')[['trend', 'seasonality', 'senoidal_cycle']]], axis=1)
X_original = copy.deepcopy(X)
X_expected = X[-TEST_ROWS:]
X = X[-(TEST_ROWS + 1):-TEST_ROWS]
#X_aux=X_aux[:-1]

series = [None] * 8
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
series[0] = utilities.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Natural_Afluente_Subsistema_Barra__data_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[1] = utilities.ReadONSEditedCSV( ONS_DIR + '/AFSum_useful.csv', 'value', mydateparser, FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[2] = utilities.ReadONSEditedCSV( ONS_DIR + '/AFSum.csv', 'value', mydateparser, FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[3] = utilities.ReadONSEditedCSV( ONS_DIR + '/Simples_Carga_de_Energia_Barra_Mês_data_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[4] = utilities.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UTE_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[5] = utilities.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UN_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[6] = utilities.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UHE_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[7] = utilities.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Armazenada_Mês_data_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]

y_original = series

norm = 'mapminmax'
if norm == 'mapstd':
    scaler = preprocessing.StandardScaler()
elif norm == 'mapstd_rob':
    scaler = preprocessing.RobustScaler()
elif norm == 'mapminmax':
    scaler = preprocessing.MinMaxScaler()

#Data scaling
X_scaler = copy.deepcopy(scaler).fit(X_original)
X_norm = pd.DataFrame(data=X_scaler.transform(X), columns=X.columns, index=X.index)

n_neurons = 106
n_folds = 8
ACTIVATION_HIDDEN_LAYER = 'relu'
ifold = 7
OPTIMIZER = 'adadelta'
NUMBER_OUTPUTS = 8
modelPath = MODELS_FOLDER + str(n_folds) + "/" + str(n_neurons) + 'neurons_' + \
            ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(ifold) \
                    +'fold_' + OPTIMIZER
model = load_model(modelPath + 'complete_best_weights.h5')
X_pred_norm = pd.DataFrame(columns=X.columns)

for month in range(TEST_ROWS):
    if month == 0:
        X_pred_norm =  pd.DataFrame(model.predict(X_norm))
    else:
        result = pd.DataFrame(model.predict(pd.DataFrame(X_pred_norm.iloc[-1,:]).T))
        X_pred_norm = X_pred_norm.append(result, ignore_index=True)
        

X_pred = pd.DataFrame(X_scaler.inverse_transform(X_pred_norm))

n_outputs = X_pred.columns.size
plt.figure()
X_pred_comp = [None] * NUMBER_OUTPUTS
for col in range(0, n_outputs):
    plt.subplot(np.ceil(n_outputs/2), 2, col + 1)
    X_pred_comp[col] = X_pred.iloc[:, col] + X_aux.iloc[-(TEST_ROWS):, 3*col: 3*(col)+3].sum(axis=1).reset_index(drop=True)
    plt.scatter(X_pred_comp[col], y_original[col][-TEST_ROWS:])
    params = np.polyfit(X_pred_comp[col], y_original[col][-TEST_ROWS:], 1)
    x_fit = params[0] * X_pred_comp[col] + params[1]
    #plt.scatter(X_pred.iloc[:, col], X_expected.iloc[:,col])
    print ( (X_pred.iloc[:, col].values - X_expected.iloc[:,col].values) / X_expected.iloc[:,col].values)
    #params = np.polyfit(X_pred.iloc[:,col], X_expected.iloc[:,col], 1)
    #x_fit = params[0] * X_pred.iloc[:, col] + params[1]
    plt.plot(X_pred_comp[col], x_fit, 'r', label='a=' + str(params[0])+ ' b=' + str(params[1]) )
    plt.legend()
    #plt.suptitle('RMSE na validação cruzada para  ' + str(n_neurons) + ' neurônios na camada intermediária')       
    #X_pred.iloc[:, col]


abc = X_pred.values - X_expected.reset_index(drop=True).values

'''
X_comp = X_scaler.inverse_transform(model.predict(X_scaler.transform(X_original)))
X_comp2 = X_scaler.inverse_transform(model.predict(X_scaler.transform(pd.DataFrame(X_original.iloc[200, :]).T)))
X_comp[-3:, 1]
X_pred.iloc[:, 1].values
'''