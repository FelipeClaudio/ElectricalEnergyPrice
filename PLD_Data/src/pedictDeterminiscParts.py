#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:59:11 2019

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
INPUT_DIR = ROOT_FOLDER + 'PLD_Data/src/Redes_Neurais/InputNN/'

PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/plots/SeriesTemporais/'

SAVE_FIG = True
SAVE_INPUT = True

input_lst = ['Total Stored Energy_tsa_decomposition.csv']
X = pd.read_csv(INPUT_DIR + 'Total Stored Energy_tsa_decomposition.csv')
