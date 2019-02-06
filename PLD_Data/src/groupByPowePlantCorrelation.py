#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 20:15:10 2019

@author: felipe
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pylab import rcParams
import seaborn as sns
import utilities as util
import trendAnalysis as tr
import statsmodels.api as sm

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

REGION = ['1 - Sudeste', '2 - Sul', '3 - Nordeste', '4 - Norte']
REGION_ABBREVIATION = ['SE', 'S', 'NE', 'N']
REGION_INDEX = 1
regionFilter = REGION[REGION_INDEX - 1]

INITIAL_YEAR = 2015
INTIAL_DATE = '2015-01-01'
FINAL_DATE = '2018-10-31'
SAVE_CORR = False
REGION_NAME = REGION_ABBREVIATION[REGION_INDEX - 1]
COR_MATRIX_FIG_NAME = REGION_NAME + '_corVec_plot.jpg'
#COR_MATRIX_FIG_NAME = 'SaoFrancisco_corVec_plot.jpg'
COR_MATRIX_DATA_NAME = REGION_NAME + '_corVec.csv'
#COR_MATRIX_DATA_NAME ='SaoFrancisco_corVec.csv'
PLOT_TITLE = 'Affluent Flow ' + REGION_NAME
#PLOT_TITLE = 'Affluent Flow São Francisco River' 
CORR_PLOT_TITLE = 'Correlation Between FLuviometric Stations - ' + REGION_NAME
#CORR_PLOT_TITLE = 'Correlation Between FLuviometric Stations - São Francisco River'
#PLOT_FIG_NAME = REGION_NAME + '_affluentFlow.jpg'
PLOT_FIG_NAME =  'SaoFrancisco_affluentFlow.jpg'
FILTER_BY_FS = False
psID = []
#psID = [156, 158, 172, 173, 175, 176, 178, 300] #São Francisco
#psID = [191, 253, 257, 270, 271, 275] #Tocantins
plt.close('all')

MIN_WINDOW_SIZE = 3
##Settings

#Extract all power stations in southeast region
hidr = pd.read_csv(MAIN_DIR + 'HIDR.csv', sep=';')
if not FILTER_BY_FS:
    hidrFiltered = hidr.loc[hidr.Sistema == regionFilter]
else:
    hidrFiltered = hidr[hidr.Posto.isin(psID)]
    
affluentFlow = pd.read_csv(MAIN_DIR + 'VAZOES_DAT.TXT', sep='\s+', header=None)
affluentFlow.columns = ['FS_ID', 'Year', 'Jan', 'Feb', 'Mar',\
                        'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']

#Extract all affluent flow by fluviometric station in a range of time
affluentFlowSFiltered= pd.merge(affluentFlow, hidrFiltered, left_on='FS_ID', right_on='Posto')
recentAFFiltered = affluentFlowSFiltered.loc[affluentFlowSFiltered.Year >= INITIAL_YEAR]
recentAFFiltered = recentAFFiltered.iloc[:, 0:21]
years = recentAFFiltered.Year.unique()
numberOfYears = years.size

#Transform data into temporal series
AFFilteredseries = recentAFFiltered.loc[recentAFFiltered.Year == INITIAL_YEAR].iloc[:, 0:14]
AFFilteredseries = AFFilteredseries.reset_index()
for year in years:
    if year != INITIAL_YEAR:
        tempAFSE = recentAFFiltered.loc[recentAFFiltered.Year == year].iloc[:, 2:14]
        tempAFSE = tempAFSE.reset_index()
        AFFilteredseries = pd.concat([AFFilteredseries, tempAFSE], axis=1)

#Get high correlated fluviometric station
AFFilteredseries = AFFilteredseries.drop(columns=['Year', 'index'])
AFFilteredseries = AFFilteredseries.iloc[:, :-2]
AFFilteredseries.set_index('FS_ID', inplace=True)
AFFilteredseriesT = AFFilteredseries.transpose()
corr = AFFilteredseriesT.corr()
ax = plt.axes()
sns.heatmap(corr, xticklabels=corr.columns, \
            yticklabels=corr.columns, cmap='Greens', ax=ax)
ax.set_title(CORR_PLOT_TITLE)
plt.show()

if SAVE_CORR:
    plt.savefig(COR_MATRIX_FIG_NAME, bbox_inches='tight')
    
#Plot all affluent flow
AFFilteredSseriesPlot = AFFilteredseriesT
months = util.GetMonthRange(INTIAL_DATE, FINAL_DATE)
AFFilteredSseriesPlot.index = months
AFFilteredSseriesPlot.plot(title=PLOT_TITLE, legend=True)
plt.ylabel('m\u00b3/s')
plt.xlabel('Year')

if SAVE_CORR:
    plt.savefig(PLOT_FIG_NAME, bbox_inches='tight')

#extend this to all fluviometric station
#THRESHOLD = 2/sqrt(N) where N is the number of samples
N = AFFilteredseries.columns.size
THRESHOLD = 2/np.sqrt(N)
corrVec = corr.iloc[1, :]

highCorVec = []
highCorVecIndex = []
for i in range(0, corrVec.size):
    tempCor = corr.iloc[i, i+1:]
    tempCor = tempCor.iloc[abs(tempCor.values) >= THRESHOLD]
    highCorVec.append(tempCor)
    highCorVecIndex.append(corrVec.index.values[i])

if SAVE_CORR:
    corr.to_csv(COR_MATRIX_DATA_NAME, sep=";")

#get total affluent flow
FStation = AFFilteredSseriesPlot.sum(axis = 1)

BEST_WINDOW_SIZE_MA = 16

AFTrend = tr.GetMovingAverage(FStation, BEST_WINDOW_SIZE_MA)
decomposition = sm.tsa.seasonal_decompose(FStation, model='additive')
AFSeasonal = decomposition.seasonal
AFResidue = FStation - AFTrend - AFSeasonal

decomposition.plot()

SAVE_FIG = False
figureName = "tsaAF_original.jpg"
if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')


figureName = "tsaAF_MA16.jpg"
plt.figure()
plt.suptitle('Temporal series analysis for afluent flow sum')

plt.subplot(4, 1, 1)
plt.plot(FStation)
plt.ylabel('Original')

plt.subplot(4, 1, 2)
plt.plot(FStation.index, AFTrend)
plt.ylabel('Trend')

plt.subplot(4, 1, 3)
plt.plot(FStation.index, AFSeasonal)
plt.ylabel('Seasonal')

plt.subplot(4, 1, 4)
plt.plot(FStation.index, AFResidue)
plt.ylabel('Residual')

if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')
    

MAX_WINDOW_SIZE = FStation.size
mseMA = np.zeros((MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) + 1)
for wSize in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
    mseMA[wSize - MIN_WINDOW_SIZE] = \
    tr.GetTrendMSEExponentialMovingAverage(FStation, wSize)

plt.figure()
plt.stem(mseMA)

mseMAS = np.zeros((MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) + 1)
for wSize in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
    mseMAS[wSize - MIN_WINDOW_SIZE] = \
    tr.GetTrendMSEExponentialMovingAverage(AFSeasonal, wSize)
    
plt.figure()
plt.title()
plt.stem(mseMAS)