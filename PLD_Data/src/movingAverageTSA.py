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
from scipy import signal

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
T_SEASONAL = 6
plt.close('all')
SAVE_FIG = False

MIN_WINDOW_SIZE = 3
bestParamString = ' W=' + str(BEST_WINDOW_SIZE_MA) + ' T=' + str(T_SEASONAL)
#Settings


mPLDSE = util.ExtractTrainTestSetFromTemporalSeries(mPLDSE, finalDate='12/2017')[0]
'''
plt.figure()
plt.plot(mPLDSE)
plt.xlabel('Month')
plt.ylabel('Price R$ MWh')
plt.title('PLD price by month')
'''

util.PlotDistribution(mPLDSE.price, xTitle='PLD Price', yTitle='Number of occurences',\
                      plotTitle='PLD distribution',\
                      filepath=PLOT_DIR+'distributionPLD.jpg',\
                      SAVE_FIGURE = SAVE_FIG)
util.FFT(mPLDSE.price, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of PLD original temporal series', figureName=PLOT_DIR+'fftPLD.jpg', \
         showPlot=True, SAVE_FIGURE=SAVE_FIG)


#pldTrend = tr.GetMovingAverage(mPLDSE.price, BEST_WINDOW_SIZE_MA)
pldTrend = tr.GetMovingAverage(mPLDSE.price, BEST_WINDOW_SIZE_MA, transitionType='smooth')
util.PlotDistribution(pldTrend, xTitle='PLD Price', yTitle='Number of occurences',\
                      plotTitle='Trend extraction distribution' + bestParamString,\
                      filepath=PLOT_DIR+'distributionTrend.jpg',\
                      SAVE_FIGURE = SAVE_FIG)
util.FFT(pldTrend, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of trend extraction' + bestParamString, figureName=PLOT_DIR+'fftTrend.jpg', \
         showPlot=True, SAVE_FIGURE=SAVE_FIG)

decomposition = sm.tsa.seasonal_decompose(mPLDSE, model='additive')
tsaSeasonal = decomposition.seasonal

bestWLinFit, minMSELinFit, mseLinFit = tr.GetTrendAnalysisByMovingAverageLinFit(mPLDSE.price, \
                                         title='MSE for trend extraction for PLD price using moving average and linear fit by window size',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         transitionType='smooth',\
                                         filepath=PLOT_DIR+'mseTrendLinFitAnalysis.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)

bestWMA,  minMSEMA, mseMA = tr.GetTrendAnalysisByMovingAverageOnly(mPLDSE.price, \
                                         title='MSE for trend extraction for PLD price using moving average by window size',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseTrendMAOnlyAnalysis.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)

bestTLinFit,  minTLinFit, mseTLinFit = tr.GetSeasonAnalysisByMovingAverageLinFit(tsaSeasonal.price, \
                                         title='MSE for seasonal extraction for PLD price using moving average and linear fit by lag time',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseSeasonalLinFitAnalysis.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)

bestTMA,  minTMa, mseTMA = tr.GetSeasonAnalysisByMovingAverageOnly(tsaSeasonal.price, \
                                         title='MSE for seasonal extraction for PLD price using moving average by lag time',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseSeasonalMAOnlyAnalysis.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)


#pldSeasonal  = tr.GetPeriodicMovingAveragePrediction(tsaSeasonal.price, T_SEASONAL)
pldSeasonal  = tr.GetPeriodicMovingAverageOnlyPrediction(tsaSeasonal.price, T_SEASONAL)
pldResidue = mPLDSE.price - pldTrend - pldSeasonal
util.PlotDistribution(pldSeasonal, xTitle='PLD Price', yTitle='Number of occurences',\
                      plotTitle='Seasonal extraction distribution' + bestParamString,\
                      filepath=PLOT_DIR+'distributionSeasonal.jpg',\
                      SAVE_FIGURE = SAVE_FIG)

util.FFT(pldSeasonal, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of seasonal extraction' + bestParamString, figureName=PLOT_DIR+'fftSeasonal.jpg', \
         showPlot=True, SAVE_FIGURE=SAVE_FIG)

util.PlotDistribution(pldResidue, xTitle='PLD Price', yTitle='Number of occurences',\
                      plotTitle='Residual extraction distribution' + bestParamString,\
                      filepath=PLOT_DIR+'distributionResidual.jpg',\
                      SAVE_FIGURE = SAVE_FIG)

util.FFT(pldResidue, xlabel='Normalized Frequency', ylabel='Normalized Magnitude', \
         title='FFT of residual extraction' + bestParamString, figureName=PLOT_DIR+'fftResidual.jpg', \
         showPlot=True, SAVE_FIGURE=SAVE_FIG)


normRes = util.NormalizeSeries(pldResidue.values.reshape(-1,1))
w0 = 0.08  # Frequency to be removed from signal (Hz)
Q = 0.1  # Quality factor
# Design notch filter
b, a = signal.iirnotch(w0, Q)
filteredResidual = signal.lfilter(b, a, normRes)
util.PlotDistribution(filteredResidual, xTitle='Normalized Price', yTitle='Ocurrences', \
                      plotTitle='Distribution for filtered residual Q='+ str(Q) + ' W0='+ str(w0) +' rad/sample' , \
                      filepath=PLOT_DIR+'filteredResidualDistribution.jpg', \
                      SAVE_FIGURE=SAVE_FIG)
util.FFT(filteredResidual, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of extracted residue after filtering Q='+ str(Q) + ' W0='+ str(w0) +' rad/sample', \
         figureName=PLOT_DIR+'filteredResidualFFT.jpg', showPlot=True,\
         SAVE_FIGURE=SAVE_FIG)

util.PlotTSA(mPLDSE, pldTrend, pldSeasonal, filteredResidual.reshape(-1,1),\
             originalPlotTitle='Original temporal series analysis for PLD prices',\
             originalPlotFileName=PLOT_DIR+'originalPLD_tsa.jpg', \
             resultPlotTitle='Result temporal series analysis for PLD prices' + bestParamString,
             resultPlotFileName=PLOT_DIR+'resultPld_tsa.jpg',
             SAVE_FIGURE=SAVE_FIG)