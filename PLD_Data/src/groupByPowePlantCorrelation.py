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

hidr = pd.read_csv(MAIN_DIR + 'HIDR.csv', sep=';')
hidrSE = hidr.loc[hidr.Sistema == '1 - Sudeste']

affluentFlow = pd.read_csv(MAIN_DIR + 'VAZOES_DAT.TXT', sep='\s+', header=None)
affluentFlow.columns = ['FS_ID', 'Year', 'Jan', 'Feb', 'Mar',\
                        'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']
affluentFlowSE = affluentFlow.loc[affluentFlow['FS_ID'] == hidrSE['Posto']]