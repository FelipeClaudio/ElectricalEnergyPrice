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
MODELS_FOLDER = ROOT_FOLDER + '/modelos_pld/t_'
#MODELS_FOLDER = ROOT_FOLDER + '/Modelos_adadelta_100epocas/'
#CLEAR_SESSION = False

N_TEST_ROWS = 3

#load trend and residue
pldTrend = pd.read_csv('pld_trend.csv', index_col=0, header=None)
pldTrend.columns = ['values']
pldSeasonal = pd.read_csv('pld_seasonal.csv', index_col=0, header=None)
pldSeasonal.columns = ['values']


#param for normalizes signal composed by residual + senoidal cycle
#w=5 / w=12
choosen_w = 3
est_amp = [94.17675855003894, 69.43507704543539, 112.68559973236269, 16.417835812860833][choosen_w]
est_freq = [0.7968374985966099, 0.8044090221275024, 0.1739844310370512, 0.23226650051018935][choosen_w]
est_mean = [-5.11368411253407, 24.509257889433268, 28.01639316680809, -6.221658471323059][choosen_w]
est_phase = [-0.2651449626303616, -0.5876011432492424, 2.7767236496885914, -0.9431117926833855][choosen_w]

t = np.arange(pldTrend.size)
senoidalCycle=est_amp*np.sin(est_freq*t+est_phase)+est_mean

# import some data to play with
X_old = pd.read_csv(INPUT_FOLDER + '/inputOld.csv').iloc[:, 1:]
X_current = pd.read_csv(INPUT_FOLDER + '/input.csv').iloc[:, 1:]
X = pd.concat([X_old, X_current], axis=1)
y_original = pd.read_csv(INPUT_FOLDER + '/output.csv').iloc[:, 1:]
y = pd.read_csv(INPUT_FOLDER + '/output_residual.csv').iloc[:, 1:]

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

n_steps = 1  
STEPS_FORECAST = n_steps
X_train = X_norm.iloc[:-(N_TEST_ROWS + STEPS_FORECAST),:]
X_test = []
if STEPS_FORECAST == 0:
    X_test = X_norm.iloc[-(N_TEST_ROWS + STEPS_FORECAST):,:]
else:
    X_test = X_norm.iloc[-(N_TEST_ROWS + STEPS_FORECAST):-STEPS_FORECAST,:]
y_train = y_norm.iloc[STEPS_FORECAST:-N_TEST_ROWS, :].reset_index(drop=True)
y_test = y_norm.iloc[-N_TEST_ROWS:, :]



# train a simple classifier
n_folds = 8
n_inits = 3
MIN_NEURONS = 90
MAX_NEURONS = 90

#partition by folder
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=0)
CVO = kf.split(X_train, y_train)
CVO = list(CVO)

mse_matrix = pd.DataFrame(columns=['fold','n_neurons', 
                                   'val_norm_rmse', 'val_rmse', 'val_std',
                                   'test_norm_rmse','test_rmse', 'test_std',
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
resultsFolds = pd.DataFrame(columns=['rmse', 'std', 'a(mean)', 'b(mean)'], index=np.arange(MAX_NEURONS) + 1)
resultsFoldsTest = pd.DataFrame(columns=['rmse', 'std', 'a(mean)', 'b(mean)'], index=np.arange(MAX_NEURONS) + 1)

for n_neurons in range(MIN_NEURONS, MAX_NEURONS + 1):
    subplotidx = 1
    if PLOT:
        plt.figure()
        
    #Read each model or best model for each folder
    for ifold in range(n_folds):
        train_id, validation_id = CVO[ifold]
        best_mse = 9999999
        current_models_folder = MODELS_FOLDER +  str(n_steps) + '/Modelos_' + str(n_folds) + '/'
        print(str(n_neurons) + ' neurons')
        if (n_steps > 1):
            modelPath = current_models_folder + str(n_neurons) + ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(ifold) \
                            +'fold_' + OPTIMIZER
        else:
            modelPath = current_models_folder + str(n_neurons) + 'neurons_'+ ACTIVATION_HIDDEN_LAYER + 'hiddenlayer_' + str(ifold) \
                            +'fold_' + OPTIMIZER
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
        params = np.polyfit(y_pred_fold_val_norm.reshape(-1), y_val_fold_norm, 1)
        mse_matrix.loc[line] =\
                      [
                              ifold, 
                              n_neurons,
                              mean_squared_error(y_val_fold_norm, y_pred_fold_val_norm),
                              np.sqrt(mean_squared_error(y_val_fold, y_pred_fold_val)),
                              np.std(y_val_fold - y_pred_fold_val),
                              mean_squared_error(y_test_norm, y_pred_test_norm),
                              np.sqrt(mean_squared_error(y_test_original, y_pred_test)),
                              np.std(y_test_original - y_pred_test),                             
                              params[0][0],
                              params[1][0]
                      ]
        
        
        #load train history
        hist = util.loadHist(modelPath + '_history.txt')
        ##min_mse = np.min(hist['val_mean_squared_error'])
        ##mse_matrix.loc[line] = [ifold, n_neurons, min_mse]
         
        
        line += 1
        del model 
        
        
        if SHOW_HISTORY:
            ax1 = plt.subplot(np.ceil(n_folds/2), 2, subplotidx)
            plt.plot(np.sqrt(hist['mean_squared_error']), label='erro no treinamento')
            plt.plot(np.sqrt(hist['val_mean_squared_error']), label='erro na validação')
            plt.legend()
            ax1.set_title('Fold ' + str(subplotidx))
            plt.suptitle('RMSE na validação cruzada para  ' + str(n_neurons) + ' neurônios na camada intermediária')
            subplotidx += 1

    #plot mse for validation set in folder
    plt.savefig(str(n_neurons) + '_convergence.jpg')
    plt.close('all')
    
    '''
    #plot mean mse by neurons number
    #plt.figure()
    plt.title('RMSE médio no conjunto de validação normalizado por número de neurônios na camada intermediária')    
    '''
    mse_mean = mse_matrix.groupby(['n_neurons']).mean().drop(columns=['fold'])
    '''
    plt.plot(np.sqrt(mse_mean['mse_val_norm']))
    plt.savefig(str(n_neurons) + '_val_norm_mse.jpg')
    plt.close('all')
    
    #plt.figure()
    plt.title('RMSE médio no conjunto de validação por número de neurônios na camada intermediária')    
    plt.plot(np.sqrt(mse_mean['mse_val']))
    plt.savefig(str(n_neurons) + '_val_mse.jpg')
    plt.close('all')
    
    #plt.figure()
    plt.title('RMSE médio no conjunto de teste normalizado por número de neurônios na camada intermediária')    
    plt.plot(np.sqrt(mse_mean['mse_test_norm']))
    plt.savefig(str(n_neurons) + '_test_norm_mse.jpg')
    plt.close('all')
    
    #plt.figure()
    plt.title('RMSE médio no conjunto de teste por número de neurônios na camada intermediária')    
    plt.plot(np.sqrt(mse_mean['mse_test']))
    plt.savefig(str(n_neurons) + '_test_mse.jpg')
    plt.close('all')
    
    
    #plot histogram
    #plt.figure()
    plt.title('RMSE médio entre os folds para o conjunto de validação desnormalizado')
    plt.hist(np.sqrt(mse_mean['mse_val']), bins=int(4*np.round(np.sqrt(mse_mean.index.size))))
    plt.savefig(str(n_neurons) + '_val_hist.jpg')
    plt.close('all')
    
    plt.title('RMSE médio entre os folds para o conjunto de validação normalizado')
    plt.hist(np.sqrt(mse_mean['mse_val_norm']), bins=int(4*np.round(np.sqrt(mse_mean.index.size))))
    plt.savefig(str(n_neurons) + '_val_hist_norm.jpg')
    plt.close('all')
    
    #plt.figure()
    plt.title('RMSE médio entre os folds para o conjunto de teste desnormalizado')
    plt.hist(np.sqrt(mse_mean['mse_test']), bins=int(4*np.round(np.sqrt(mse_mean.index.size))))
    plt.savefig(str(n_neurons) + '_test_hist.jpg')
    plt.close('all')
    
    plt.title('RMSE médio entre os folds para o conjunto de teste normalizado')
    plt.hist(np.sqrt(mse_mean['mse_test_norm']), bins=int(4*np.round(np.sqrt(mse_mean.index.size))))
    plt.savefig(str(n_neurons) + '_test_hist_norm.jpg')
    plt.close('all')
    '''
    
    #scatter plot
    #plt.figure()

    model = load_model(modelPath + 'complete_best_weights.h5')
                    
    #mse_comp.loc[n_neurons]= [y_original.iloc[-N_TEST_ROWS:,:] - y_pred_comp]
    #mse_comp.plot(title='Erro absoluto no conjunto de teste')
    idxs = np.arange(n_neurons) + 1
    
    resultsFolds.loc[n_neurons] = [
                                mse_mean.loc[n_neurons]['val_rmse'], 
                                mse_mean.loc[n_neurons]['val_std'],
                                mse_mean.loc[n_neurons]['a'],
                                mse_mean.loc[n_neurons]['b']
                             ]    

                                    
    plt.title('Scatter Série prevista X residual para o conjunto completo')
    y_comp = model.predict(X_norm)
    '''
    plt.scatter(y_comp, y_norm)
    plt.xlabel('Série prevista')
    plt.ylabel('Série original')
    params = np.polyfit(y_comp.reshape(-1), y_norm, 1)
    y_fit = params[0] * y_comp + params[1]
    plt.plot(y_comp, y_fit, 'r', label='a=' + str(params[0])+ ' b=' + str(params[1]) )
    plt.legend()
    plt.savefig(str(n_neurons) + '_residual_scatter.jpg')
    plt.close('all')
    '''
    results.loc[n_neurons] = [
            np.sqrt(mean_squared_error(y_original.iloc[-N_TEST_ROWS:,:], y_comp[-N_TEST_ROWS:])), \
           np.std(y_original.iloc[-N_TEST_ROWS:,:] - y_comp[-N_TEST_ROWS:]), params[0], params[1]]
    
    
    y_pred_test = y_comp[-N_TEST_ROWS:]
    params_test = np.polyfit(y_pred_test.reshape(-1), y_test, 1)
    resultsFoldsTest.loc[n_neurons] = [
            np.sqrt(mean_squared_error(y_pred_test, y_test)), \
            np.std(y_pred_test - y_test), params_test[0], params_test[1]]
    del model
    print(str(n_neurons) + " neurons")
    
#results.plot(title="RMSE absoluto no conjunto de validação por número de neurônios na camada intermediária")
x_val = np.arange(results.iloc[:,0].size)
plt.plot(x_val, results['rmse'], label='rmse')
plt.title('RMSE absoluto no conjunto de validação por número de neurônios na camada intermediária')
plt.legend()
plt.savefig('rmse_val_set.jpg')
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.close('all')
'''
resultsFoldsTest.plot(title="RMSE absoluto no conjunto de teste por número de neurônios na camada intermediária")
plt.savefig('rmse_test_set.jpg')
plt.close('all')
'''
x_test = np.arange(resultsFoldsTest.iloc[:,0].size)
plt.plot(x_test, resultsFoldsTest['rmse'], label='rmse')
plt.title('RMSE absoluto no conjunto de teste por número de neurônios na camada intermediária')
plt.legend()
plt.savefig('rmse_test_set.jpg')
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.close('all')
    
resultsFolds.to_csv('./resultsFold.csv')
resultsFoldsTest.to_csv('./resultsFold_Test.csv')

NUM_ELEMENTS = 28
y_original = y_original.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):]
y = y.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):]
reco = y_scaler.inverse_transform(y_comp[STEPS_FORECAST:, :]) + pldTrend.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):].values + \
pldSeasonal.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):].values + \
senoidalCycle[-(NUM_ELEMENTS - STEPS_FORECAST):].reshape(-1, 1)
#plt.plot(t, senoidalCycle)


reco2 = pldTrend.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):].values + pldSeasonal.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):].values # + \
#senoidalCycle[-28:].reshape(-1, 1)
x = [1, 2, 3]
plt.plot(x, pldTrend.iloc[-3:].values +pldSeasonal.iloc[-3:].values + senoidalCycle[-3:].reshape(-1, 1), 'r')
plt.plot(x, y_original.iloc[-3:].values, 'b')
plt.plot(x, reco[-3:], 'g')

'''
x2 = np.arange(NUM_ELEMENTS-1)
plt.figure()
plt.plot(x2, pldTrend.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):].values + \
         pldSeasonal.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):].values + \
         senoidalCycle[-(NUM_ELEMENTS - STEPS_FORECAST):].reshape(-1, 1), 'r')
plt.plot(x2, y_original.iloc[-(NUM_ELEMENTS - STEPS_FORECAST):].values, 'b')
plt.plot(x2, reco[-NUM_ELEMENTS:], 'g')
'''
y2 = y_original - y #- senoidalCycle[-28:].reshape(-1, 1)
t2 = np.arange(y2.size)


recoCSV = pd.DataFrame(data=reco, columns=['value'])
recoCSV.to_csv('./reco.csv')

y_error = y_original - reco
y_error.to_csv('./yError.csv')

'''
plt.figure()
plt.plot(t2, reco2, 'r')
plt.plot(t2, y2, 'b')
plt.show()
'''


'''
plt.figure()
plt.plot(y_comp, 'r')
plt.plot(y_norm, 'b')

plt.figure()
plt.scatter(reco, y_original)
plt.title('previsto X original para ' + str(MAX_NEURONS) + " neurônios")
plt.xlabel('previsto')
plt.ylabel('original')
params = np.polyfit(reco.reshape(-1), y_original, 1)
y_fit = params[0] * reco + params[1]
plt.plot(reco, y_fit, label='a=' + str(params[0])+ ' b=' + str(params[1]))
plt.legend()
plt.show()

np.max(np.abs(y_original - reco))
np.std(y_original - reco)
np.min(np.abs(y_original - reco))
'''


y_size = y_original.size
y_deterministico = pldTrend.iloc[-y_size:].values + \
         pldSeasonal.iloc[-y_size:].values + \
         senoidalCycle[-y_size:].reshape(-1, 1)
plt.figure()
plt.plot(y_original.index, reco, 'r', label='sinal com resíduo previsto')
plt.plot(y_original.index, y_original, 'b', label='original')
plt.plot(y_original.index, y_deterministico, 'y', label='sinal sem resíduo')
plt.errorbar(y_original.index, reco, yerr=resultsFolds.loc[MAX_NEURONS]['std'], fmt='r', ecolor='black')
plt.legend()
plt.xlabel('Amostra')
plt.ylabel('Valor PLD')
plt.title('PLD médio mensal previsto X real para o mês atual '+ str(MAX_NEURONS) + ' neurônios')
plt.show()
plt.savefig('sinal_completo_t' + str(n_steps) + '.jpg')

'''
error_series = np.sqrt(np.square(y_original - reco))
np.mean(error_series)
'''