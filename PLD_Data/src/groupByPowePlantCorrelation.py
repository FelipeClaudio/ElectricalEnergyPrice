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

##Settings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

ROOT_FOLDER  = 'D:/Sistemas GIT/ElectricalEnergyPrice'

#loading PLD data
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Outubro_2018'
MAIN_DIR += '/10_out18_RV0_logENA_Mer_d_preco_m_0/'

REGION = ['1 - Sudeste', '2 - Sul', '3 - Nordeste', '4 - Norte']
regionFilter = REGION[1 - 1]

INITIAL_YEAR = 2015
INTIAL_DATE = '2015-01-01'
FINAL_DATE = '2018-10-31'
SAVE_CORR = True
REGION_NAME = 'SE'
COR_MATRIX_FIG_NAME = REGION_NAME + '_corVec_plot.jpg'
COR_MATRIX_DATA_NAME = REGION_NAME + '_corVec.csv'
PLOT_TITLE = 'Affluent Flow ' + REGION_NAME
CORR_PLOT_TITLE = 'Correlation Between FLuviometric Stations - ' + REGION_NAME
PLOT_FIG_NAME = REGION_NAME + '_affluentFlow.jpg'
##Settings

#Extract all power stations in southeast region
hidr = pd.read_csv(MAIN_DIR + 'HIDR.csv', sep=';')
hidrSE = hidr.loc[hidr.Sistema == regionFilter]

affluentFlow = pd.read_csv(MAIN_DIR + 'VAZOES_DAT.TXT', sep='\s+', header=None)
affluentFlow.columns = ['FS_ID', 'Year', 'Jan', 'Feb', 'Mar',\
                        'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']

#Extract all affluent flow by fluviometric station in a range of time
affluentFlowSE = pd.merge(affluentFlow, hidrSE, left_on='FS_ID', right_on='Posto')
recentAFSE = affluentFlowSE.loc[affluentFlowSE.Year >= INITIAL_YEAR]
recentAFSE = recentAFSE.iloc[:, 0:21]
years = recentAFSE.Year.unique()
numberOfYears = years.size

#Transform data into temporal series
AFSEseries = recentAFSE.loc[recentAFSE.Year == INITIAL_YEAR].iloc[:, 0:14]
AFSEseries = AFSEseries.reset_index()
for year in years:
    if year != INITIAL_YEAR:
        tempAFSE = recentAFSE.loc[recentAFSE.Year == year].iloc[:, 2:14]
        tempAFSE = tempAFSE.reset_index()
        AFSEseries = pd.concat([AFSEseries, tempAFSE], axis=1)

#Get high correlated fluviometric station
AFSEseries = AFSEseries.drop(columns=['Year', 'index'])
AFSEseries = AFSEseries.iloc[:, :-2]
AFSEseries.set_index('FS_ID', inplace=True)
AFSEseriesT = AFSEseries.transpose()
corr = AFSEseriesT.corr()
ax = plt.axes()
sns.heatmap(corr, xticklabels=corr.columns, \
            yticklabels=corr.columns, cmap='Greens', ax=ax)
ax.set_title(CORR_PLOT_TITLE)
plt.show()

if SAVE_CORR:
    plt.savefig(COR_MATRIX_FIG_NAME, bbox_inches='tight')
    
#Plot all affluent flow
AFSESseriesPlot = AFSEseriesT
months = util.GetMonthRange(INTIAL_DATE, FINAL_DATE)
AFSESseriesPlot.index = months
AFSESseriesPlot.plot(title=PLOT_TITLE, legend=False)
plt.ylabel('m\u00b3/s')
plt.xlabel('Year')

if SAVE_CORR:
    plt.savefig(PLOT_FIG_NAME, bbox_inches='tight')

#extend this to all fluviometric station
#THRESHOLD = 2/sqrt(N) where N is the number of samples
N = AFSEseries.columns.size
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
FStation = AFSEseries.sum(axis = 1)
