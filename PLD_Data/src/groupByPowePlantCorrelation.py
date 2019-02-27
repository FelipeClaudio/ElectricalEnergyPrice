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

PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/plots/SeriesTemporais/Vazoes/'

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
hidrBase=pd.read_csv(MAIN_DIR + 'HIDR_BASE.csv')
REMOVE_MIN_VOLUME=True
SAVE_FIG=False
#For total volume
#BEST_WINDOW_SIZE_MA = 14
#BEST_T = 12

#For useful volume
BEST_WINDOW_SIZE_MA = 14
T_SEASONAL =6

bestParamString = ' W=' + str(BEST_WINDOW_SIZE_MA) + ' T=' + str(T_SEASONAL)
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

def NoNegative(x):
    if x < 0:
        return 0
    return x

if REMOVE_MIN_VOLUME:
    for index, row in hidrBase.iterrows():
        AFFilteredseries.loc[AFFilteredseries.index == row['Posto']] -= float(row['VolMin'])
        
AFFilteredseries = AFFilteredseries.applymap(NoNegative)

AFFilteredseriesT = AFFilteredseries.transpose()
corr = AFFilteredseriesT.corr()
ax = plt.axes()
sns.heatmap(corr, xticklabels=corr.columns, \
            yticklabels=corr.columns, cmap='Greens', ax=ax)
ax.set_title(CORR_PLOT_TITLE)
plt.show()

if SAVE_CORR:
    plt.savefig(PLOT_DIR+COR_MATRIX_FIG_NAME, bbox_inches='tight')
    
#Plot all affluent flow
AFFilteredSseriesPlot = AFFilteredseriesT
months = util.GetMonthRange(INTIAL_DATE, FINAL_DATE)
AFFilteredSseriesPlot.index = months
AFFilteredSseriesPlot.plot(title=PLOT_TITLE, legend=True)
plt.ylabel('m\u00b3/s')
plt.xlabel('Year')

if SAVE_CORR:
    plt.savefig(PLOT_DIR+PLOT_FIG_NAME, bbox_inches='tight')

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
FStation.to_csv('AFSum_useful.csv')

util.PlotDistribution(FStation, xTitle='Affluent flow sum SE', yTitle='Number of occurences',\
                      plotTitle='Affluent flow sum SE distribution' + bestParamString,\
                      filepath=PLOT_DIR+'distributionOriginal_AF.jpg',\
                      SAVE_FIGURE = SAVE_FIG)
util.FFT(FStation, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of affluent fLow sum SE' + bestParamString, figureName=PLOT_DIR+'fftOriginal_AF.jpg', \
         showPlot=True, SAVE_FIGURE=SAVE_FIG)

AFTrend = tr.GetMovingAverage(FStation, BEST_WINDOW_SIZE_MA, transitionType='smooth') 
#AFTrend = tr.GetMovingAverage(FStation, BEST_WINDOW_SIZE_MA, tr.GetTrendMSEMovingAverageOnly)
decomposition = sm.tsa.seasonal_decompose(FStation, model='additive')

util.PlotDistribution(AFTrend, xTitle='Affluent flow sum SE trend extraction', yTitle='Number of occurences',\
                      plotTitle='Affluent flow sum SE trend extraction distribution' + bestParamString,\
                      filepath=PLOT_DIR+'distributionTrend_AF.jpg',\
                      SAVE_FIGURE = SAVE_FIG)
util.FFT(AFTrend, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of Affluent fLow sum SE trend extraction' + bestParamString, figureName=PLOT_DIR+'fftTrend_AF.jpg', \
         showPlot=True, SAVE_FIGURE=SAVE_FIG)

AFSeasonal = decomposition.seasonal


util.PlotDistribution(AFSeasonal, xTitle='Affluent flow sum SE seasonal extraction', yTitle='Number of occurences',\
                      plotTitle='Affluent flow sum SE seasonal extraction distribution' + bestParamString,\
                      filepath=PLOT_DIR+'distributionSeasonal_AF.jpg',\
                      SAVE_FIGURE = SAVE_FIG)
util.FFT(AFSeasonal, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of Affluent fLow sum SE seasonal extraction' + bestParamString, figureName=PLOT_DIR+'fftSeasonal_AF.jpg', \
         showPlot=True, SAVE_FIGURE=SAVE_FIG)
 

bestWLinFit, minMSELinFit, mseLinFit = tr.GetTrendAnalysisByMovingAverageLinFit(FStation, \
                                         title='MSE for trend extraction for PLD price using moving average and linear fit by window size',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         transitionType='smooth',\
                                         filepath=PLOT_DIR+'mseTrendLinFitAnalysis_AF_SUM.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)

bestWMA,  minMSEMA, mseMA = tr.GetTrendAnalysisByMovingAverageOnly(FStation, \
                                         title='MSE for trend extraction for PLD price using moving average by window size',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseTrendMAOnlyAnalysis_AF_SUM.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)

AFSeasonal = pd.Series(AFSeasonal)
bestTLinFit,  minTLinFit, mseTLinFit = tr.GetSeasonAnalysisByMovingAverageLinFit(AFSeasonal, \
                                         title='MSE for seasonal extraction for PLD price using moving average and linear fit by lag time',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseSeasonalLinFitAnalysis_AF_SUM.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)

bestTMA,  minTMa, mseTMA = tr.GetSeasonAnalysisByMovingAverageOnly(AFSeasonal, \
                                         title='MSE for seasonal extraction for PLD price using moving average by lag time',\
                                         MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                         filepath=PLOT_DIR+'mseSeasonalMAOnlyAnalysis_AF_SUM.jpg',\
                                         SAVE_FIGURE = SAVE_FIG)



AFSeasonalBest = tr.GetPeriodicMovingAverageOnlyPrediction(AFSeasonal, T_SEASONAL)
AFResidue = FStation - AFTrend - AFSeasonalBest

normRes = util.NormalizeSeries(AFResidue.values.reshape(-1,1))
util.PlotDistribution(normRes, xTitle='Normalized affluent flow', yTitle='Ocurrences', \
                      plotTitle='Distribution for residual extraction' + bestParamString, \
                      filepath=PLOT_DIR+'ResidualDistribution_AF.jpg', \
                      SAVE_FIGURE=SAVE_FIG)
util.FFT(normRes, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of extracted residual of affluent flow sum' + bestParamString, \
         figureName=PLOT_DIR+'ResidualFFT_AF.jpg', showPlot=True,\
         SAVE_FIGURE=SAVE_FIG)


w0 = 0.08  # Frequency to be removed from signal (Hz)
Q = 0.1  # Quality factor
# Design notch filter
b, a = signal.iirnotch(w0, Q)
filteredResidual = signal.lfilter(b, a, normRes)
util.PlotDistribution(filteredResidual, xTitle='Normalized affluent flow', yTitle='Ocurrences', \
                      plotTitle='Distribution for filtered residual Q='+ str(Q) + ' W0='+ str(w0) +' rad/sample' , \
                      filepath=PLOT_DIR+'filteredResidualDistribution_AF.jpg', \
                      SAVE_FIGURE=SAVE_FIG)
util.FFT(filteredResidual, xlabel='Normalized Frequency', ylabel='Magnitude', \
         title='FFT of filtered extracted residual of affluent flow sum Q='+ str(Q) + ' W0='+ str(w0) +' rad/sample', \
         figureName=PLOT_DIR+'filteredResidualFFT_AF.jpg', showPlot=True,\
         SAVE_FIGURE=SAVE_FIG)

util.PlotTSA(FStation, AFTrend, AFSeasonalBest.reshape(-1,1), filteredResidual.reshape(-1,1),\
             originalPlotTitle='Original temporal series analysis of affluent flow sum SE',\
             originalPlotFileName=PLOT_DIR+'originalAF_tsa.jpg', \
             resultPlotTitle='Result temporal series analysis for afluent flow sum se',
             resultPlotFileName=PLOT_DIR+'resultAF_tsa.jpg',
             SAVE_FIGURE=SAVE_FIG)