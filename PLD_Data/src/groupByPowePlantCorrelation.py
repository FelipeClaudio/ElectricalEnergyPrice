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
import statsmodels.api as sm
import trendAnalysis as tr


warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

ROOT_FOLDER  = '/home/felipe/Materias/TCC'

#loading PLD data
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Outubro_2018'
MAIN_DIR += '/10_out18_RV0_logENA_Mer_d_preco_m_0/'

#Extract all power stations in southeast region
hidr = pd.read_csv(MAIN_DIR + 'HIDR.csv', sep=';')
hidrSE = hidr.loc[hidr.Sistema == '1 - Sudeste']

affluentFlow = pd.read_csv(MAIN_DIR + 'VAZOES_DAT.TXT', sep='\s+', header=None)
affluentFlow.columns = ['FS_ID', 'Year', 'Jan', 'Feb', 'Mar',\
                        'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']

#Extract all affluent flow by fluviometric station in a range of time
affluentFlowSE = pd.merge(affluentFlow, hidrSE, left_on='FS_ID', right_on='Posto')
INITIAL_DATE = 2015
recentAFSE = affluentFlowSE.loc[affluentFlowSE.Year >= INITIAL_DATE]
recentAFSE = recentAFSE.iloc[:, 0:21]
years = recentAFSE.Year.unique()
numberOfYears = years.size

'''
concatAFSE = pd.DataFrame(np.zeros( shape=(recentAFSE.FS_ID.unique().size, \
                                             (recentAFSE.columns.size * numberOfYears)\
                         - (numberOfYears - 1))))
'''

#Transform data into temporal series
AFSEseries = recentAFSE.loc[recentAFSE.Year == INITIAL_DATE].iloc[:, 0:14]
AFSEseries = AFSEseries.reset_index()
for year in years:
    if year != INITIAL_DATE:
        tempAFSE = recentAFSE.loc[recentAFSE.Year == year].iloc[:, 2:14]
        tempAFSE = tempAFSE.reset_index()
        AFSEseries = pd.concat([AFSEseries, tempAFSE], axis=1)


#Get high correlated fluviometric station
AFSEseries = AFSEseries.drop(columns=['Year', 'index'])
AFSEseries = AFSEseries.iloc[:, :-2]
AFSEseries.set_index('FS_ID', inplace=True)
AFSEseriesT = AFSEseries.transpose()
cor = AFSEseriesT.corr()
plt.matshow(cor)
plt.show()

#extend this to all fluviometric station
THRESHOLD = 0.9
corrVec = cor.iloc[1, :]

highCorVec = []
for i in range(0, corrVec.size):
    tempCor = cor.iloc[i, i+1:]
    tempCor = tempCor.iloc[tempCor.values >= THRESHOLD]
    highCorVec.append(tempCor)