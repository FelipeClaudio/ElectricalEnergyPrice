#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:02:13 2019

@author: felipe
"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pylab import rcParams
import statsmodels.api as sm
import trendAnalysis as tr
import utilities as util
import locale
import seaborn as sns
import math

locale.setlocale(locale.LC_TIME, "en_US.UTF-8") 

##Settings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
'''
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
'''

ROOT_FOLDER  = '/home/felipe/Materias/TCC/'
#loading PLD data
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Outubro_2018'
MAIN_DIR += '/10_out18_RV0_logENA_Mer_d_preco_m_0/'
ONS_DIR = ROOT_FOLDER + 'PLD_Data/ONS_DATA'
mydateparser = lambda x: pd.datetime.strptime(x, "%m/%Y")
meanPLD = pd.read_csv(MAIN_DIR + 'PLD_medio.csv', \
                      parse_dates=['Mês'], sep="\\s+",\
                      date_parser=mydateparser)


PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/plots/SeriesTemporais/'
INITIAL_DATE = '01/2015'
FINAL_DATE = '12/2018'
mPLDSE = meanPLD[['Mês', 'SE/CO']]
mPLDSE.columns = ['month', 'price']
mPLDSE.set_index('month', inplace=True)
mPLDSE = mPLDSE.sort_index()
mPLDSE = mPLDSE.loc[mPLDSE.index >= INITIAL_DATE]
mPLDSE = mPLDSE.loc[mPLDSE.index <= FINAL_DATE]

mPLDSE.index.inferred_freq

BEST_WINDOW_SIZE_MA = 16
T_SEASONAL = 12
plt.close('all')
SAVE_FIG = True
FINAL_DATE = '12/2018'

language = util.language.PT.value
suffix = "_pt"
textVec = {
            "correlation_matrix": ["Correlation between ONS DATA and PLD between in traning set",\
                                   "Correlação entre os dados fornecidos pela ONS e o PLD"],
            "TotalEnergy": ["Total Stored Energy","Energia Total Armazenda"],
            "UHEEnergy": ["UHE Generated Energy", "Energia Gerada por UHEs"],
            "UNEnergy": ["UN Generated Energy", "Energia gerada por UNs"],
            "UTEEnergy": ["UTE Energy", "Energia gerada por UTEs"],
            "SolarEnergy": ["Solar Generated Energy", "Energia gerada por usinas solares"],
            "WindEnergy": ["Wind Generated Energy", "Energia gerada por usinas eólicas"],
            "LoadEnergy": ["Load Energia", "Energia na carga"],
            "ENA": ["ENA", "ENA"],
            "AFsum": ["Affluent Flow Sum", "Soma das vazões afluentes"],
            "AFSumUseful": ["Useful Affluent Flow Sum", "Soma das vazões afluentes úteis"]
        }
language = util.language.PT.value
#Settings


totalStoredEnergy = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Armazenada_Mês_data_editado.csv', textVec["TotalEnergy"][language], FINAL_DATE=FINAL_DATE)[0]
uheGeneratedEnergy = util.ReadONSEditedCSV (ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UHE_editado.csv', textVec["UHEEnergy"][language] ,FINAL_DATE=FINAL_DATE)[0]
unGeneratedEnergy = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UN_editado.csv', textVec["UNEnergy"][language], FINAL_DATE=FINAL_DATE)[0]
uteGeneratedEnergy = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_UTE_editado.csv', textVec["UTEEnergy"][language], FINAL_DATE=FINAL_DATE)[0]
solarGeneratedEnergy = util.ReadONSEditedCSV(ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_solar_editado.csv', textVec["SolarEnergy"][language], FINAL_DATE=FINAL_DATE)[0]
windGeneratedEnergy = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Geração_de_Energia_Barra_Mês_data_eolica_editado.csv', textVec["WindEnergy"][language], FINAL_DATE=FINAL_DATE)[0]
loadEnergy = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Carga_de_Energia_Barra_Mês_data_editado.csv', textVec["LoadEnergy"][language], FINAL_DATE=FINAL_DATE)[0]
ena = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Natural_Afluente_Subsistema_Barra__data_editado.csv', textVec["ENA"][language], FINAL_DATE=FINAL_DATE)[0]
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
afSum = util.ReadONSEditedCSV( ONS_DIR + '/AFSum.csv', textVec["AFsum"][language], mydateparser, FINAL_DATE=FINAL_DATE)[0]
afSumUseful = util.ReadONSEditedCSV( ONS_DIR + '/AFSum_useful.csv', textVec["AFSumUseful"][language], mydateparser, FINAL_DATE=FINAL_DATE)[0]

cor = pd.concat([mPLDSE, totalStoredEnergy], axis=1)
cor = pd.concat([cor, uheGeneratedEnergy], axis=1)
cor = pd.concat([cor, uteGeneratedEnergy], axis=1)
cor = pd.concat([cor, solarGeneratedEnergy], axis=1)
cor = pd.concat([cor, windGeneratedEnergy], axis=1)
cor = pd.concat([cor, loadEnergy], axis=1)
cor = pd.concat([cor, ena], axis=1)
cor = pd.concat([cor, afSum], axis=1)
cor = pd.concat([cor, afSumUseful], axis=1)

ax = plt.axes()
correlation = cor.corr()
correlation.to_csv('corr_ons.csv')
sns.heatmap(correlation, xticklabels=correlation.columns, \
            yticklabels=correlation.columns, cmap='RdBu', ax=ax,\
            annot=True, fmt=".2f")

#sns.scatterplot(correlation)
ax.set_title(textVec["correlation_matrix"][language])
plt.show()
if SAVE_FIG:
    plt.savefig("corr_plot"+ suffix + ".jpg", bbox_inches='tight')
