#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 08:13:03 2019

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
from FourierSeriesMinimizer import FourierSeriesMinimizer
import utilities as util


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
#loading PLD data
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Outubro_2018'
MAIN_DIR += '/10_out18_RV0_logENA_Mer_d_preco_m_0/'
mydateparser = lambda x: pd.datetime.strptime(x, "%m/%Y")
meanPLD = pd.read_csv(MAIN_DIR + 'PLD_medio.csv', \
                      parse_dates=['Mês'], sep="\\s+",\
                      date_parser=mydateparser)


PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/plots/SeriesTemporais/'
INITIAL_DATE = '01/2015'
mPLDSE = meanPLD[['Mês', 'SE/CO']]
mPLDSE.columns = ['month', 'price']
mPLDSE.set_index('month', inplace=True)
mPLDSE = mPLDSE.sort_index()
mPLDSE = mPLDSE.loc[mPLDSE.index >= INITIAL_DATE]

mPLDSE.index.inferred_freq

BEST_WINDOW_SIZE_MA = 16
plt.close('all')
SAVE_FIG = True
#Settings

pldTrend = tr.GetMovingAverage(mPLDSE.price, BEST_WINDOW_SIZE_MA)
decomposition = sm.tsa.seasonal_decompose(mPLDSE, model='additive')
tsaSeasonal = decomposition.seasonal

MIN_WINDOW_SIZE = 1
MAX_WINDOW_SIZE = tsaSeasonal.size
mseMA = np.zeros((MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) + 1)
for wSize in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
    mseMA[wSize - MIN_WINDOW_SIZE] = \
    tr.GetTrendMSEMovingAveragePeriodic(tsaSeasonal, wSize)

pldSeasonal = []
pldResidue = mPLDSE.price - pldTrend - pldSeasonal.price

decomposition.plot()
figureName = "tsa_original.jpg"
if SAVE_FIG:
    plt.savefig(PLOT_DIR + figureName, bbox_inches='tight')


figureName = "tsa_MA16.jpg"
plt.figure()
plt.suptitle('Temporal series analysis for PLD prices')

plt.subplot(4, 1, 1)
plt.plot(mPLDSE)
plt.ylabel('Original')

plt.subplot(4, 1, 2)
plt.plot(mPLDSE.index, pldTrend)
plt.ylabel('Trend')

plt.subplot(4, 1, 3)
plt.plot(mPLDSE.index, pldSeasonal)
plt.ylabel('Seasonal')

plt.subplot(4, 1, 4)
plt.plot(mPLDSE.index, pldResidue)
plt.ylabel('Residual')

if SAVE_FIG:
    plt.savefig(PLOT_DIR + figureName, bbox_inches='tight')
    

