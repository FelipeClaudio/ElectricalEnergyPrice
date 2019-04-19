#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:17:18 2019

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
ONS_DIR = ROOT_FOLDER + 'PLD_Data/ONS_DATA'
INPUT_DIR = ROOT_FOLDER + 'PLD_Data/src/inputNN/'

#loading PLD data
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Outubro_2018'
MAIN_DIR += '/10_out18_RV0_logENA_Mer_d_preco_m_0/'
mydateparser = lambda x: pd.datetime.strptime(x, "%m/%Y")
meanPLD = pd.read_csv('PLD_medio.csv', \
                      parse_dates=['Mês'], sep="\\s+",\
                      date_parser=mydateparser)



FINAL_DATE = '12/2018'
PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/plots/SeriesTemporais/'
INITIAL_DATE = '01/2015'
mPLDSE = meanPLD[['Mês', 'SE/CO']]
mPLDSE.columns = ['month', 'price']
mPLDSE.set_index('month', inplace=True)
mPLDSE = mPLDSE.sort_index()
mPLDSE = util.ExtractTrainTestSetFromTemporalSeries(mPLDSE, initialDate=INITIAL_DATE,\
                                                    finalDate=FINAL_DATE)[0]

SAVE_FIG = False
SAVE_INPUT = True
#Settings
ax = autocorrelation_plot(mPLDSE)
if SAVE_FIG:
    plt.savefig(ROOT_FOLDER+'PLD_Data/src/plots/autocorrelation_PLD_train.jpg', bbox_inches='tight')
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.05, 0.95, 'sdfad', transform=ax.transAxes, fontsize=14,\
#        verticalalignment='top', bbox=props)
ax.legend(['99% confidence', '95% confidence'])
ax.set_title('Autocorrelation of PLD price')

totalStoredEnergyOriginal = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Armazenada_Mês_data_editado.csv', 'Total Stored Energy', FINAL_DATE=FINAL_DATE)[0]
uheGeneratedEnergyOriginal = util.ReadONSEditedCSV (ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UHE_editado.csv', 'UHE Generated Energy', FINAL_DATE=FINAL_DATE)[0]
unGeneratedEnergyOriginal = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UN_editado.csv', 'UN Generated Energy', FINAL_DATE=FINAL_DATE)[0]
uteGeneratedEnergyOriginal = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UTE_editado.csv', 'UTE Generated Energy', FINAL_DATE=FINAL_DATE)[0]
solarGeneratedEnergyOriginal = util.ReadONSEditedCSV(ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_solar_editado.csv', 'Solar Generated Energy', FINAL_DATE=FINAL_DATE)[0]
windGeneratedEnergyOriginal = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_eolica_editado.csv', 'Wind Generated Energy', FINAL_DATE=FINAL_DATE)[0]
loadEnergyOriginal = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Carga_de_Energia_Barra_Mês_data_editado.csv', 'Load Energy', FINAL_DATE=FINAL_DATE)[0]
enaOriginal = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Natural_Afluente_Subsistema_Barra__data_editado.csv', 'ENA', FINAL_DATE=FINAL_DATE)[0]
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
afSumOriginal = util.ReadONSEditedCSV( ONS_DIR + '/AFSum.csv', 'Afluent Flow Sum', mydateparser, FINAL_DATE=FINAL_DATE)[0]
afSumUsefulOriginal = util.ReadONSEditedCSV( ONS_DIR + '/AFSum_useful.csv', 'Useful Afluent Flow Sum', mydateparser, FINAL_DATE=FINAL_DATE)[0]

numberOfLags = mPLDSE.size - max(util.GetDefaultMask()) - 1
NUMBER_OF_VARIABLES = 9

for lag in range(0, numberOfLags):
    shiftLeft = lag
    totalStoredEnergy = util.GetFilteredSeriesByMask(totalStoredEnergyOriginal, shiftLeft=shiftLeft)
    uheGeneratedEnergy = util.GetFilteredSeriesByMask(uheGeneratedEnergyOriginal, shiftLeft=shiftLeft)
    unGeneratedEnergy = util.GetFilteredSeriesByMask(unGeneratedEnergyOriginal, shiftLeft=shiftLeft)
    #uteGeneratedEnergy = util.GetFilteredSeriesByMask(uteGeneratedEnergyOriginal, shiftLeft=shiftLeft)
    solarGeneratedEnergy = util.GetFilteredSeriesByMask(solarGeneratedEnergyOriginal, shiftLeft=shiftLeft)
    windGeneratedEnergy = util.GetFilteredSeriesByMask(windGeneratedEnergyOriginal, shiftLeft=shiftLeft)
    loadEnergy = util.GetFilteredSeriesByMask(loadEnergyOriginal, shiftLeft=shiftLeft)
    ena = util.GetFilteredSeriesByMask(enaOriginal, shiftLeft=shiftLeft)
    afSum = util.GetFilteredSeriesByMask(afSumOriginal, shiftLeft=shiftLeft)
    afSumUseful = util.GetFilteredSeriesByMask(afSumUsefulOriginal, shiftLeft=shiftLeft)  
    
    inputDataOld = util.TransposeAndSetColumnNames(totalStoredEnergy[1], 'tEn')
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(uheGeneratedEnergy[1],  'gUHE')], axis=1)
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(unGeneratedEnergy[1],  'gUN')], axis=1)
    #inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(uteGeneratedEnergy[1],  'gUTE')], axis=1)
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(solarGeneratedEnergy[1],  'gUSolar')], axis=1)
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(windGeneratedEnergy[1],  'gUWind')], axis=1)
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(loadEnergy[1],  'load')], axis=1)
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(ena[1],  'ENA')], axis=1)
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(afSum[1],  'AfSum')], axis=1)
    inputDataOld = pd.concat([inputDataOld, util.TransposeAndSetColumnNames(afSumUseful[1],  'uAfSum')], axis=1)
    
    inputData = totalStoredEnergy[0]
    inputData = pd.concat([inputData, uheGeneratedEnergy[0]], axis=1)
    inputData = pd.concat([inputData, unGeneratedEnergy[0]], axis=1)
    #inputData = pd.concat([inputData, uteGeneratedEnergy[0]], axis=1)
    inputData = pd.concat([inputData, solarGeneratedEnergy[0]], axis=1)
    inputData = pd.concat([inputData, windGeneratedEnergy[0]], axis=1)
    inputData = pd.concat([inputData, loadEnergy[0]], axis=1)
    inputData = pd.concat([inputData, ena[0]], axis=1)
    inputData = pd.concat([inputData, afSum[0]], axis=1)
    inputData = pd.concat([inputData, afSumUseful[0]], axis=1)
    inputData = inputData.reset_index(drop=True)
    #inputData.columns = ['tEn', 'gUHE', 'gUN', 'gUTE', 'gUSolar', 'gUWind', 'load', 'ENA', 'AfSum', 'uAfSum']
    inputData.columns = ['tEn', 'gUHE', 'gUN', 'gUSolar', 'gUWind', 'load', 'ENA', 'AfSum', 'uAfSum']
    
    if lag == 0:
        finalInput = inputData
        finalInputOld = inputDataOld
    else:
        finalInput = pd.concat([finalInput, inputData])
        finalInputOld = pd.concat([finalInputOld, inputDataOld])

#Order by input by temporal time
#Older predicions comes first
finalInput = finalInput.reset_index(drop=True)
#Recover dates related to each row
numElem = finalInput.index.size
dateRange = util.GetMonthRange(INITIAL_DATE, FINAL_DATE, '%m/%Y')
indexDates = dateRange[-numElem:]
finalInput.index = indexDates

finalInputOld = finalInputOld.reset_index(drop=True)

finalOutput = mPLDSE
finalOutput = finalOutput.reset_index(drop=True)
finalOutput = finalOutput.iloc[-numberOfLags:,:]
if SAVE_INPUT:
    finalInput.to_csv('input.csv')
    finalInputOld.to_csv('inputOld.csv')
    finalOutput.to_csv('output.csv')
