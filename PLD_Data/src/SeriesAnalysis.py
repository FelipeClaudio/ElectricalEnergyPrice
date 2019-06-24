#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:45:24 2019

@author: felipe
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pandas.plotting import autocorrelation_plot
import numpy as np
from pylab import rcParams
import statsmodels.api as sm
import trendAnalysis as tr
import utilities as util
from scipy import signal
import locale
locale.setlocale(locale.LC_TIME, "en_US.UTF-8") 

##Settings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

ROOT_FOLDER  = '/home/felipe/Materias/TCC/'
ONS_DIR = ROOT_FOLDER + 'PLD_Data/ONS_DATA_COMPLETE'
INPUT_DIR = ROOT_FOLDER + 'PLD_Data/src/inputNN/'
PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/'

#loading PLD data
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Dezembro_2018'
MAIN_DIR += '/12_dez18_RV0_logENA_Mer_PEN_d_preco_m_0/'
mydateparser = lambda x: pd.datetime.strptime(x, "%m/%Y")
meanPLD = pd.read_csv('PLD_medio.csv', \
                      parse_dates=['Mês'], sep="\\s+",\
                      date_parser=mydateparser)



FINAL_DATE = '12/2018'
PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/plots/SeriesTemporais/'
INITIAL_DATE = '01/2002'
mPLDSE = meanPLD[['Mês', 'SE/CO']]
mPLDSE.columns = ['month', 'price']
mPLDSE.set_index('month', inplace=True)
mPLDSE = mPLDSE.sort_index()
mPLDSE = util.ExtractTrainTestSetFromTemporalSeries(mPLDSE, initialDate=INITIAL_DATE,\
                                                    finalDate=FINAL_DATE)[0]

SAVE_FIG = False
SAVE_INPUT = True
#Settings
'''
ax = autocorrelation_plot(mPLDSE)
if SAVE_FIG:
    plt.savefig(ROOT_FOLDER+'PLD_Data/src/plots/autocorrelation_PLD_train.jpg', bbox_inches='tight')

'''
titles=['Total Stored Energy', 'UHE Generated Energy', 'UN Generated Energy', \
        'UTE Generated Energy','Load Energy', 'ENA', \
        'Afluent Flow Sum', 'Useful Afluent Flow Sum']
'''
titles=['Total Stored Energy', 'UHE Generated Energy', 'UN Generated Energy', \
        'UTE Generated Energy', 'Load Energy', 'ENA', \
        'Afluent Flow Sum', 'Useful Afluent Flow Sum', \
        'Solar Generated Energy', 'Wind Generated Energy' ]
'''
series = [None] * 8
series[0] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Armazenada_Mês_data_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[1] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UHE_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[2] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UN_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[3] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UTE_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[4] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Carga_de_Energia_Barra_Mês_data_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[5] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Natural_Afluente_Subsistema_Barra__data_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
series[6] = util.ReadONSEditedCSV( ONS_DIR + '/AFSum.csv', 'value', mydateparser, FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
series[7] = util.ReadONSEditedCSV( ONS_DIR + '/AFSum_useful.csv', 'value', mydateparser, FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]
#series[8] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_eolica_editado.csv', 'value', FINAL_DATE=FINAL_DATE)[0]
#series[4] = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_solar_editado.csv', 'value', FINAL_DATE=FINAL_DATE, INITIAL_DATE=INITIAL_DATE)[0]

analysisMatrix =  pd.DataFrame(columns=['bestWLinFit', 'minMSELinFit', 'mseLinFit', 'bestTMA', 'minTMA', 'mseTMA'])
index = 0

MAX_W = 14
MAX_T = 6
w0 = 0.2
Q = 0.1


for title in titles:
    result = pd.DataFrame(columns=['trend', 'seasonality', 'senoidal_cycle', 'residual'])
    analysisMatrix.loc[index] = tr.GetTemporalSeriesCompleteAnalysis(series[index], PLOT_DIR, title=title, SAVE_FIGURE=True)
    if analysisMatrix.loc[index]['bestWLinFit'] > MAX_W:
        W = MAX_W
    else:
        W = analysisMatrix.loc[index]['bestWLinFit']
        
    if analysisMatrix.loc[index]['bestTMA'] > MAX_T:
        T = MAX_T
    else:
        T = analysisMatrix.loc[index]['bestTMA']
        
    trend = tr.GetMovingAverage(series[index].value, W, transitionType='smooth')
    decomposition = sm.tsa.seasonal_decompose(series[index], model='additive')
    seasonal = tr.GetPeriodicMovingAverageOnlyPrediction(decomposition.seasonal.value, T)
    senoidal_cycle = tr.EstimateSenoidalCycle(seasonal)
    residual = tr.GetResidualExtraction(series[index], W, T, w0, Q)
    #result.loc[index] = [trend, seasonal, senoidal_cycle, residual]
    result['trend']=trend
    result['seasonality']=seasonal
    result['senoidal_cycle']=senoidal_cycle
    result['residual']=residual
    result.to_csv(title + '_tsa_decomposition.csv')
    del result
    index = index + 1