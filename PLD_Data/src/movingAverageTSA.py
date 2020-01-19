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
from copy import deepcopy

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
#MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Outubro_2018'
#MAIN_DIR += '/10_out18_RV0_logENA_Mer_d_preco_m_0/'
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
bestWString =  ' W=' + str(BEST_WINDOW_SIZE_MA)
bestParamString = ' W=' + str(BEST_WINDOW_SIZE_MA) + ' T=' + str(T_SEASONAL)

textVec = {
           "pld_price": ["PLD Price", "Preço do PLD"],
           "n_occur": ["Number of occurences", "Número de ocorrências"],
           "pld_dist": ["PLD Distribution", "Distribuição do PLD"],
           "norm_freq": ["Normalized Frequency", "Frequência Normalizada"],
           "fft_orig": ["FFT of PLD original temporal series", "FFT da série original do PLD"],
           "original_fft_dist": ["FFT and distribution of PLD", "FFT e distribuição do PLD"],
           "trend_ext": ["FFT of PLD's extracted trend","FFT da tedência extraída do PLD"],
           "trend_fft_dist": ["FFT and distribution of PLD trend", "FFT e distribuição da tendência do PLD"],
           "trend_mse_linfit": ["MSE for trend extraction for PLD price using moving average and linear fit by window size", \
                                "MSE para a extração de tendência do PLD usando média móvel e fit linear por tamanho da janela"],
           "trend_mse_ma": ["MSE for trend extraction for PLD price using moving average", \
                                "MSE para a extração de tendência do PLD usando média móvel"],
           "seasonal_mse_linfit": ["MSE for seasonal extraction for PLD price using linear regression by lag time", \
                                "MSE para a extração da sazonalidade do PLD usando regressão linear por lags temporais"],
           "seasonal_mse_ma": ["MSE for seasonal extraction for PLD price using moving average ", \
                                "MSE para a extração de sazonalidade do PLD usando média móvel"],
           "seasonal_dist": ["Distribution of PLD's extracted seasonality","Ditribuição da sazonalidade extraída do PLD"],
           "seasonal_fft": ["FFT of extracted seasonality","FFT da sazonalidade extraída"],
           "seasonal_fft_dist": ["FFT and distribution of PLD seasonality", "FFT e distribuição da sazonalidade do PLD"],
           "norm_price": ["Normalized PLD price", "Preço do PLD normalizado"],
           "res_ext": ["FFT of extracted residue + senoidal cycle","FFT do resíduo + ciclos senoidais extraídos"],
           "res_fft_dist": ["FFT and distribution of PLD residue + senoidal cycle", "FFT e distribuição do resíduo + ciclo senoidal do PLD"],
           "filt_res_dist": ["Distribution of filtered residual using", "Distribuição para a série residual filtrada usando "],
           "norm_mag": ["Normalized Magnitude","Magnitude Normalizada"],
           "filt_res_fft": ["FFT of filtered residual using", "FFT do sinal residual filtrado usando"],
           "filt_res_fft_dist": ["FFT and Distribution of filtered residual using", "FFT e distribuição do sinal residual filtrado usando"],
           "rad_sample": ["rads/sample", "rads/amostra"],
           "result_tsa": ["Result of temporal series analysis", "Resultado da análise de séries temporais"],
           "original_plot": ["Original result for temporal series analysis", "Resultado original para análise de séries temporais"],
           "pld_plot": ["Mean PLD price by month", "PLD médio por mês"]
        }
language = util.language.PT.value
suffix = "_pt"
FINAL_DATE = '12/2018'
w0 = 0.10869 # Frequency to be removed from signal (Hz)
Q = 0.01  # Quality factor
filterParams = ' Q='+ str(Q) + ' W0='+ str(w0) + ' ' +  textVec["rad_sample"][language]
#Settings


mPLDSE = util.ExtractTrainTestSetFromTemporalSeries(mPLDSE, finalDate=FINAL_DATE)[0]
mPLDSE_PT = deepcopy(mPLDSE)
mPLDSE_PT.columns=['preço']
mPLDSE_PT.index.rename('mês', inplace=True)
mPLDSE_PT.plot()
plt.title(textVec["pld_plot"][language])
plt.ylabel('preço')
if SAVE_FIG:
    plt.savefig(PLOT_DIR  + "pld" + suffix + ".jpg", bbox_inches='tight')

'''
plt.figure()
plt.plot(mPLDSE)
plt.xlabel('Month')
plt.ylabel('Price R$ MWh')
plt.title('PLD price by month')
'''


figureName = PLOT_DIR + 'fftAndDistribution_original'+ suffix +'.jpg'
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
util.PlotDistribution(mPLDSE.price, xTitle=textVec["pld_price"][language], yTitle=textVec["n_occur"][language],\
                      plotTitle=textVec["pld_dist"][language],\
                      filepath=PLOT_DIR+'distributionPLD' + suffix + '.jpg',\
                      ax=ax1, SAVE_FIGURE = SAVE_FIG)
ax2= plt.subplot(1, 2, 2)
util.FFT(mPLDSE.price, xlabel=textVec["norm_freq"][language], ylabel='Magnitude', \
         title=textVec["fft_orig"][language], figureName=PLOT_DIR+'fftPLD'+suffix+'.jpg', \
         ax=ax2, showPlot=True, SAVE_FIGURE=SAVE_FIG)
fig.suptitle(textVec["original_fft_dist"][language])

if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')


figureName = PLOT_DIR + 'fftAndDistribution_trend' + suffix + '.jpg'
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
#pldTrend = tr.GetMovingAverage(mPLDSE.price, BEST_WINDOW_SIZE_MA)
pldTrend = tr.GetMovingAverage(mPLDSE.price, BEST_WINDOW_SIZE_MA, transitionType='smooth')
util.PlotDistribution(pldTrend, xTitle=textVec["pld_price"][language], yTitle=textVec["n_occur"][language],\
                      plotTitle=textVec["trend_ext"][language] + bestWString,\
                      filepath=PLOT_DIR+'distributionTrend'+ suffix +'.jpg',\
                      ax=ax1, SAVE_FIGURE = SAVE_FIG)

ax2= plt.subplot(1, 2, 2)
util.FFT(pldTrend, xlabel=textVec["norm_freq"][language], ylabel='Magnitude', \
         title=textVec["trend_ext"][language] + bestWString, figureName=PLOT_DIR+'fftTrend'+ suffix +'.jpg', \
         ax=ax2, showPlot=True, SAVE_FIGURE=SAVE_FIG)
fig.suptitle(textVec["trend_fft_dist"][language] + bestWString)

if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')

decomposition = sm.tsa.seasonal_decompose(mPLDSE, model='additive')

'''
tsaSeasonalprice = tr.seasonal_mean(mPLDSE.price)
tsaSeasonal = pd.DataFrame(data = {'price': tsaSeasonalprice})
tsaSeasonal.index = mPLDSE.index
'''

tsaSeasonal = decomposition.seasonal

figureName = PLOT_DIR + 'mseComparation_trend'+ suffix +'.jpg'
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
bestWLinFit, minMSELinFit, mseLinFit = tr.GetTrendAnalysisByMovingAverageLinFit(mPLDSE.price, \
                                         title=textVec["trend_mse_linfit"][language],\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         transitionType='smooth',\
                                         filepath=PLOT_DIR+'mseTrendLinFitAnalysis'+ suffix +'.jpg',\
                                         ax=ax1, SAVE_FIGURE=SAVE_FIG)

ax2 = plt.subplot(2, 1, 2)
bestWMA,  minMSEMA, mseMA = tr.GetTrendAnalysisByMovingAverageOnly(mPLDSE.price, \
                                         title=textVec["trend_mse_ma"][language],\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseTrendMAOnlyAnalysis'+ suffix +'.jpg',\
                                         ax=ax2, SAVE_FIGURE=SAVE_FIG)
plt.subplots_adjust(hspace=0.3)

if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')


figureName = PLOT_DIR + 'mseComparation_tseasonal'+ suffix +'.jpg'
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
bestTLinFit,  minTLinFit, mseTLinFit = tr.GetSeasonAnalysisByMovingAverageLinFit(tsaSeasonal.price, \
                                         title=textVec["seasonal_mse_linfit"][language],\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseSeasonalLinFitAnalysis.jpg',\
                                         ax=ax1, SAVE_FIGURE = SAVE_FIG)

ax2 = plt.subplot(2, 1, 2)
bestTMA,  minTMa, mseTMA = tr.GetSeasonAnalysisByMovingAverageOnly(tsaSeasonal.price, \
                                         title=textVec["seasonal_mse_ma"][language],\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseSeasonalMAOnlyAnalysis.jpg',\
                                         ax=ax2, SAVE_FIGURE = SAVE_FIG)

plt.subplots_adjust(hspace=0.3)
if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')

figureName = PLOT_DIR + 'fftAndDistribution_seasonal'+ suffix +'.jpg'
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
#pldSeasonal  = tr.GetPeriodicMovingAveragePrediction(tsaSeasonal.price, T_SEASONAL)
pldSeasonal  = tr.GetPeriodicMovingAverageOnlyPrediction(tsaSeasonal.price, T_SEASONAL)
pldResidue = mPLDSE.price - pldTrend - pldSeasonal
util.PlotDistribution(pldSeasonal, xTitle=textVec["pld_price"][language], yTitle=textVec["n_occur"][language],\
                      plotTitle=textVec["seasonal_dist"][language] + bestParamString,\
                      filepath=PLOT_DIR+'distributionSeasonal'+ suffix +'.jpg',\
                      ax=ax1, SAVE_FIGURE=SAVE_FIG)

ax2= plt.subplot(1, 2, 2)
util.FFT(pldSeasonal, xlabel=textVec["norm_freq"][language], ylabel=textVec["norm_mag"][language], \
         title=textVec["seasonal_fft"][language] + bestParamString, figureName=PLOT_DIR+'fftSeasonal.jpg', \
         ax=ax2, showPlot=True, SAVE_FIGURE=SAVE_FIG)

fig.suptitle(textVec["seasonal_fft_dist"][language] + bestParamString)

if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')


figureName = PLOT_DIR + 'fftAndDistribution_residual'+ suffix +'.jpg'
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
util.PlotDistribution(pldResidue, xTitle=textVec["pld_price"][language], yTitle=textVec["n_occur"][language],\
                      plotTitle=textVec["filt_res_dist"][language] + bestParamString,\
                      filepath=PLOT_DIR+'distributionResidual'+ suffix +'.jpg',\
                      ax=ax1, SAVE_FIGURE=SAVE_FIG)

ax2= plt.subplot(1, 2, 2)
util.FFT(pldResidue, xlabel=textVec["norm_freq"][language], ylabel=textVec["norm_mag"][language], \
         title='FFT of residual extraction' + bestParamString, figureName=PLOT_DIR+'fftResidual'+ suffix +'.jpg', \
         ax= ax2, showPlot=True, SAVE_FIGURE=SAVE_FIG)

fig.suptitle(textVec["res_fft_dist"][language] )

if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')

normRes = util.NormalizeSeries(pldResidue.values.reshape(-1,1))
# Design notch filter
b, a = signal.iirnotch(w0, Q)
filteredResidual = signal.lfilter(b, a, normRes)

figureName = PLOT_DIR + 'fftAndDistribution_residualFiltered'+ suffix +'.jpg'
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
util.PlotDistribution(filteredResidual, xTitle=textVec["norm_price"][language], yTitle=textVec["n_occur"][language], \
                      plotTitle=textVec["filt_res_dist"][language] + bestParamString + filterParams , \
                      filepath=PLOT_DIR+'filteredResidualDistribution.'+ suffix +'.jpg', \
                      ax=ax1, SAVE_FIGURE=SAVE_FIG)

ax2= plt.subplot(1, 2, 2)
util.FFT(filteredResidual, xlabel=textVec["norm_freq"][language], ylabel='Magnitude', \
         title=textVec["filt_res_fft"][language] + bestParamString + filterParams, \
         figureName=PLOT_DIR+'filteredResidualFFT'+ suffix +'.jpg', showPlot=True,\
         ax=ax2, SAVE_FIGURE=SAVE_FIG)

fig.suptitle(textVec["filt_res_fft_dist"][language] + bestParamString + filterParams)
if SAVE_FIG:
    plt.savefig(figureName, bbox_inches='tight')

util.PlotTSA(mPLDSE, pldTrend, pldSeasonal, filteredResidual.reshape(-1,1),\
             originalPlotTitle=textVec["original_plot"][language],\
             originalPlotFileName=PLOT_DIR+'originalPLD_tsa'+ suffix +'.jpg', \
             resultPlotTitle=textVec["result_tsa"][language] + bestParamString + filterParams,
             resultPlotFileName=PLOT_DIR+'resultPld_tsa'+ suffix +'.jpg',
             SAVE_FIGURE=SAVE_FIG)

results = pd.DataFrame(columns=['Trend', 'Seazonal', 'Senoidal Cycle', 'Residual'], index=mPLDSE.index)