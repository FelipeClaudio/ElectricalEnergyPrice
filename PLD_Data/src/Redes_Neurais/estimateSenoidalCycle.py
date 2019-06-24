#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:26:10 2019

@author: felipe
"""

import warnings
import pandas as pd
import matplotlib
import numpy as np
from pylab import rcParams
import statsmodels.api as sm
import trendAnalysis as tr
import utilities as util
from scipy import signal
import numpy as np
from scipy.optimize import leastsq
import pylab as plt
import sys
sys.path.append('/home/felipe/Materias/TCC/PLD_Data/src/')
import trendAnalysis as tr
'''
N = 1000 # number of data points
t = np.linspace(0, 4*np.pi, N)
f = 1.15247 # Optional!! Advised not to use
data = 3.0*np.sin(f*t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise
'''
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
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Dezembro_2018'
MAIN_DIR += '/12_dez18_RV0_logENA_Mer_PEN_d_preco_m_0/'
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

BEST_WINDOW_SIZE_MA = 12
T_SEASONAL = 6
plt.close('all')
SAVE_FIG = True

MIN_WINDOW_SIZE = 3
bestParamString = ' W=' + str(BEST_WINDOW_SIZE_MA) + ' T=' + str(T_SEASONAL)
#Settings


decomposition = sm.tsa.seasonal_decompose(mPLDSE, model='additive')
tsaSeasonal = decomposition.seasonal
pldTrend = tr.GetMovingAverage(mPLDSE.price, BEST_WINDOW_SIZE_MA, transitionType='smooth')
pldSeasonal  = tr.GetPeriodicMovingAverageOnlyPrediction(tsaSeasonal.price, T_SEASONAL)
pldResidue = mPLDSE.price - pldTrend - pldSeasonal
t = np.arange(pldResidue.size)
w0 = 0.2  # Frequency to be removed from signal (Hz)
Q = 0.1  # Quality factor
# Design notch filter
b, a = signal.iirnotch(w0, Q)
filteredResidual = signal.lfilter(b, a, pldResidue)

guess_mean = np.mean(pldResidue)
guess_std = 3*np.std(pldResidue)/(2**0.5)/(2**0.5)
guess_phase = 0
guess_freq = 0.2
guess_amp = pldResidue.max()

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_std*np.sin(t + guess_phase) + guess_mean

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.sin(x[1]*t +x[2]) + x[3] - pldResidue
est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

# recreate the fitted curve using the optimized parameters
data_fit_original = est_amp*np.sin(est_freq*t+est_phase) + est_mean
# recreate the fitted curve using the optimized parameters

fine_t = np.arange(0,max(t),0.1)
data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean

plt.plot(t, pldResidue, '.')
plt.plot(fine_t, data_fit, label='after fitting')
plt.legend()
plt.show()


plt.figure()
plt.plot(t, pldResidue, 'r', label='original residue')
plt.plot(t, filteredResidual + data_fit_original, label='estimimated residue: \nwo_est=' + str(est_freq)\
         + '\namp_est=' + str(est_amp) + '\nphi_est='+str(est_phase) + '\nmean_est='+str(est_mean))
plt.legend()
plt.show()
plt.savefig('realXestimated_residueandsenoidalcycles.jpg')

pd.Series(pldTrend).to_csv('pld_trend.csv')
pd.Series(pldSeasonal).to_csv('pld_seasonal.csv')

util.FFT(pldResidue.iloc[:-12], 'a', 'b', 'c', 'd', showPlot=True)