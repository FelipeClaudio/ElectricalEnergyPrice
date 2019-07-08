#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:21:59 2019

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
#MODELS_FOLDER = ROOT_FOLDER + '/Modelos_adadelta_100epocas/'
#CLEAR_SESSION = False

N_TEST_ROWS = 24

#load trend and residue
pldTrend = pd.read_csv('pld_trend.csv', index_col=0, header=None)
pldTrend.columns = ['values']
pldSeasonal = pd.read_csv('pld_seasonal.csv', index_col=0, header=None)
pldSeasonal.columns = ['values']


# import some data to play with
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
X_new=X[:-1]
X_aux=X_aux[:-1]
#y = pd.read_csv(INPUT_FOLDER + '/output.csv').iloc[:, 1:]
y = X[1:]
X = X_new

FINAL_DATE = '12/2018'
ROOT_FOLDER  = '/home/felipe/Materias/TCC/'
ONS_DIR = ROOT_FOLDER + 'PLD_Data/ONS_DATA_COMPLETE'
INITIAL_DATE = '01/2002'
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

'''
for i in range(0, X.columns.size):
    if i == 0:
        y_original = X.iloc[:,i] + X_aux.iloc[:, 3*i:3*i +2].sum(axis=1)
    else:
        temp = X.iloc[:,i] + X_aux.iloc[:, 3*i:(3*i+2)].sum(axis=1) 
        y_original = pd.concat([y_original, temp], axis=1)
'''

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

#partition by folder
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=0)
CVO = kf.split(X_train, y_train)
CVO = list(CVO)
    
mse_matrix = pd.DataFrame(columns=['output','fold','n_neurons', 
                                   'mse_val_norm', 'mse_val',
                                   'mse_test_norm','mse_test',
                                   'val_rmse', 'val_std',
                                   'a', 'b'])

line = 0
hasModel = False
OPTIMIZER = 'adadelta'
ACTIVATION_HIDDEN_LAYER = 'relu'
line = 0
PLOT = False
SHOW_HISTORY = True
READ_BEST_MODEL = True
resultsFolds = pd.DataFrame(columns=['output', 'rmse', 'std', 'a(mean)', 'b(mean)'], index=np.arange(MAX_NEURONS) + 1)
histVec = [None] * MAX_NEURONS
NUMBER_OUTPUTS = 8

results_res = pd.DataFrame(columns=['n_neurons', 'output', 'a_comp', 'b_comp'])

for n_neurons in range(MIN_NEURONS, MAX_NEURONS + 1):
    subplotidx = 1
    if PLOT:
        plt.figure()
        
    #Read each model or best model for each folder
    for ifold in range(n_folds):
        train_id, validation_id = CVO[ifold]
        modelPath = MODELS_FOLDER + str(n_folds) + "/" + str(n_neurons) + 'neurons_' + \
                    ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(ifold) \
                    +'fold_' + OPTIMIZER
        print(modelPath)
        K.clear_session()
        
        #choose between best model or last model
        if READ_BEST_MODEL:
            model = load_model(modelPath + 'best_weights.h5')
        else:
            model = load_model(modelPath + '.h5')


        y_val_fold_norm = y_train.iloc[validation_id]
        y_val_fold =  y_scaler.inverse_transform(y_val_fold_norm)
        y_pred_fold_val_norm = model.predict(X_train.iloc[validation_id])
        y_pred_fold_val = y_scaler.inverse_transform(y_pred_fold_val_norm)

        #get test data error
        y_test_norm = y_test
        y_pred_test_norm = model.predict(X_test)
        y_pred_test = y_scaler.inverse_transform(y_pred_test_norm)
        y_test_original = y_scaler.inverse_transform(y_test)
        
        
        for output in range(NUMBER_OUTPUTS):
            params = np.polyfit(y_pred_fold_val_norm[:,output].reshape(-1), y_val_fold_norm.iloc[:,output], 1)
            mse_matrix.loc[line] =\
                          [
                                  output,
                                  ifold, 
                                  n_neurons,
                                  mean_squared_error(y_val_fold_norm.iloc[:,output], y_pred_fold_val_norm[:, output]),
                                  mean_squared_error(y_val_fold[:,output], y_pred_fold_val[:,output]),
                                  mean_squared_error(y_test_norm.iloc[:,output], y_pred_test_norm[:,output]),
                                  mean_squared_error(y_test_original[:,output], y_pred_test[:,output]),
                                  np.sqrt(mean_squared_error(y_val_fold[:,output], y_pred_fold_val[:,output])),
                                  np.std(y_val_fold[:,output] - y_pred_fold_val[:,output]),
                                  params[0],
                                  params[1]
                          ]
            line += 1
        
        #load train history
        hist = util.loadHist(modelPath + '_history.txt')
         
        '''
        y_test_norm = y_test
        y_pred_test_norm = model.predict(X_test)
        y_pred_test = y_scaler.inverse_transform(y_pred_test_norm)
        y_test_original = y_scaler.inverse_transform(y_test)

        
        
        '''
        
        line += 1
        del model 
        
        
        if SHOW_HISTORY:
            ax1 = plt.subplot(np.ceil(n_folds/2), 2, subplotidx)
            plt.plot(np.sqrt(hist['mean_squared_error']), label='train error')
            plt.plot(np.sqrt(hist['val_mean_squared_error']), label='validation error')
            plt.legend()
            ax1.set_title('Fold ' + str(subplotidx))
            plt.suptitle('RMSE na validação cruzada para  ' + str(n_neurons) + ' neurônios na camada intermediária')
            subplotidx += 1
        
     
    #plot mse for validation set in folder
    plt.savefig(str(n_neurons) + '_convergence.jpg')
    plt.close('all')
    
    model = load_model(modelPath + 'complete_best_weights.h5')
    y_comp = y_scaler.inverse_transform(model.predict(X_norm))

    plot_x = np.arange(X.index.size)
    y_pred_comp = [None] * NUMBER_OUTPUTS
    for out_idx in range(0, NUMBER_OUTPUTS):
        #all parts of decomposition in order to get original signal
        y_pred_comp[out_idx] = y_comp[:, out_idx] + X_aux.iloc[:, 3*out_idx: 3*(out_idx)+3].sum(axis=1)
        plt.subplot(round(NUMBER_OUTPUTS/2), 2, out_idx+1)   
        plt.plot(plot_x, y_original[out_idx][1:], 'r', label='Original')
        plt.plot(plot_x, y_pred_comp[out_idx], 'b', label='Previsto')
        plt.legend()

    plt.savefig(str(n_neurons) + '_previsoes.jpg')
    plt.close()
    
    for out_idx in range(0, NUMBER_OUTPUTS):
        #all parts of decomposition in order to get original signal
        plt.subplot(round(NUMBER_OUTPUTS/2), 2, out_idx+1)   
        plt.plot(plot_x, y.iloc[:, out_idx], 'r', label='Original')
        plt.plot(plot_x, y_comp[:, out_idx], 'b', label='Previsto')
        plt.legend()

    plt.savefig(str(n_neurons) + '_previsoes_residuos.jpg')
    plt.close()
    
    for out_idx in range(0, NUMBER_OUTPUTS):
        plt.subplot(round(NUMBER_OUTPUTS/2), 2, out_idx+1)
        #all parts of decomposition in order to get original signal
        y_pred_comp[out_idx] = y_comp[:, out_idx] + X_aux.iloc[:, 3*out_idx: 3*(out_idx)+3].sum(axis=1)
        plt.scatter(y_pred_comp[out_idx], y_original[out_idx][1:])
        params = np.polyfit(y_pred_comp[out_idx], y_original[out_idx][1:], 1)
        y_fit = params[0] * y_pred_comp[out_idx] + params[1]
        plt.plot(y_pred_comp[out_idx], y_fit, 'r', label='a=' + str(params[0])+ ' b=' + str(params[1]) )
        plt.legend()   
    plt.savefig(str(n_neurons) + '_fit.jpg')
    plt.close()
    
    for out_idx in range(0, NUMBER_OUTPUTS):
        plt.subplot(round(NUMBER_OUTPUTS/2), 2, out_idx+1)
        #all parts of decomposition in order to get original signal
        plt.scatter(y_comp[:, out_idx], y.iloc[:, out_idx])
        params = np.polyfit(y.iloc[:, out_idx], y_comp[:, out_idx], 1)
        y_fit = params[0] * y_comp[:, out_idx] + params[1]
        plt.plot(y_comp[:, out_idx], y_fit, 'r', label='a=' + str(params[0])+ ' b=' + str(params[1]) )
        plt.legend()
        results_res.loc[out_idx % NUMBER_OUTPUTS + (n_neurons -1) * NUMBER_OUTPUTS]  = [n_neurons, out_idx, params[0], params[1]]
    plt.savefig(str(n_neurons) + '_fit_residuos.jpg')
    plt.close()
    
    #del model
    del hist
    print(str(n_neurons) + " neurons")

    #salvar o A, B, std e média obtidos em um csv
    
mse_mean = mse_matrix.groupby(['n_neurons', 'output']).mean().drop(columns=['fold'])
mse_mean.to_csv('./resultsFold_input.csv')
results_res.to_csv('./results_input_residual.csv')


for output in range(0, NUMBER_OUTPUTS):
    plt.subplot(round(NUMBER_OUTPUTS/2), 2, output+1)   
    plt.suptitle('RMSE médio no conjunto de validação por número de neurônios na camada intermediária')    
    plt.plot(mse_mean.iloc[mse_mean.index.get_level_values('output') == output]['mse_val'].values)

plt.savefig(str(n_neurons) + '_val_mse.jpg')
plt.close('all')