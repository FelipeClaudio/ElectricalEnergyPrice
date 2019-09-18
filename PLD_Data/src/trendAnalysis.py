# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 22:16:53 2018

@author: felip
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
from scipy import signal
from scipy.optimize import leastsq

# MatPlotlib
import matplotlib.pyplot as plt
from matplotlib import pylab
from pandas.core.nanops import nanmean as pd_nanmean

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


#OTHERS
def predictDeterministcSeries(x, n_steps, windowSize):
    y_array = []
    
    total_offsets = 5
    for offset in range(0, total_offsets):
        size = n_steps + offset + 2
        y = np.zeros(size)
        for i in range(0, size):
            if (i < (2 + offset)):
                y[i] = x[i]
            elif (i >= (2 + offset) and i < windowSize ):
                fit = np.polyfit(np.arange(0, x.size - 1), x[:-1], 1)
                fit_fn = np.poly1d(fit)
                y[i] = fit_fn(i)
            else:
                y[i] = np.mean(y[i-windowSize:i])
        y_array.append(y)
        
    y_comp = y_array[-1]
    for y_temp in y_array:
        size = y_temp.size - 1
        y_comp[size] = y_temp[size]
        
    return y_comp


def PredictFirstWindowPoints(y, windowSize):
    yWithNoMean = y[0:windowSize - 1]
    yTemp = [0] * windowSize
    for i in range(0, windowSize):
        if i <= 1:
            yTemp[i] = yWithNoMean[i]
        else:
            fit = np.polyfit(np.arange(0, i), yWithNoMean[0:i], 1)
            fit_fn = np.poly1d(fit)
            yTemp[i] = fit_fn(i)
    return yTemp

def PredictFirstWindowPointsMA(y, windowSize):
    yWithNoMean = y[0:windowSize - 1]
    yTemp = [0] * windowSize
    for i in range(0, windowSize):
        if i <= 1:
            yTemp[i] = yWithNoMean[i]
        else:
            yTemp[i] = np.mean(yWithNoMean[0:i])
            
    return yTemp     
    
    
def ParameterVariationAnalysis(y, paramFunction, MIN_WINDOW_SIZE,
                               MAX_WINDOW_SIZE, transitionType=None):   
    mse = np.zeros((MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) + 1)
    for param in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
        mse[param - MIN_WINDOW_SIZE] = paramFunction(y, param, transitionType)
        
    return mse


def StemPlotErrorAnalysis(y, MIN_WINDOW_SIZE, xTitle, yTitle, figTitle, filepath, ax=None, SAVE_FIGURE=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    y = np.array(y)
    #Remove zero elements in right side of plot
    nZeroElems = np.nonzero(y)[0]
    y = y[0:np.max(nZeroElems)]
    #Add zeros into the beginnig of the plot in order to facilitate visualization
    y = np.concatenate((np.zeros(MIN_WINDOW_SIZE), y))

    ax.stem(y)
    ax.set_title(figTitle)
    ax.set_ylabel(yTitle)
    ax.set_xlabel(xTitle)
    if SAVE_FIGURE:
        plt.savefig(filepath, bbox_inches='tight')    

def PlotPolyFit(x, y, originalX, coef, FILE_PATH, figureName, showPlot = False, SAVE_FIGURE = False):    
    #uses poly1d to get Y values given coefficients
    poly = np.poly1d(coef)
    yPredicted = poly(x)
    mse = mean_squared_error(y, yPredicted)
    rmse = sqrt(mse)
    
    if showPlot:
        plt.figure()
        plt.plot(originalX, y,'o', originalX, yPredicted)
        plt.title(figureName)
        plt.xlabel('Month')
        plt.ylabel('PLD price')
        ax = plt.gca()
        ax.set_facecolor((0.898, 0.898, 0.898))
        
        #add text box with info about RMSE and MSE
        textstr = '\n'.join((
            r'MSE=%.2f' % (mse, ),
            r'RMSE=%.2f' % (rmse, )
            ))
    
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    if SAVE_FIGURE:
        plt.savefig(FILE_PATH , bbox_inches='tight')
    return [mse]

def PlotErrorFunction(x, errorValues, INITIAL_ORDER, FINAL_ORDER, filepath, \
                        figureName, xLabel='Order', \
                        yLabel='Value', SAVE_FIGURE=False):
    '''
    INPUT:
        x:              array with horizontal axis info
        errorValues:    array with error for a given prediction
        INITIAL_ORDER:  order of the lowest polynomial used in error array
        FINAL_ORDER:    order of the highest polynomial used in error array
        FILE_PATH:      folder where file will be saved
        figureName;     title shown in figure
        xLabel:         label for x axis in plot
        SAVE_FIGURE:    enables saving plot feature
    OUTUPUT:

    '''  
    plotAxis = np.arange(INITIAL_ORDER, FINAL_ORDER)
    plt.figure()
    plt.plot(plotAxis, errorValues)
    plt.title(figureName)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if (SAVE_FIGURE == True):
        plt.savefig(filepath, bbox_inches='tight')
        
def SelectBestParam(mseVec, MIN_ORDER):
    mseBefore = 0
    mseAfter = 0
    minMse = np.max(mseVec)
    minIndex = 0
    
    for i in range(1, mseVec.size - 1):
        mseBefore = mseVec[i-1]
        mseAfter = mseVec[i+1]
        msePoint = mseVec[i]
        
        if (msePoint < mseAfter) and \
        (msePoint < mseBefore) and \
        (msePoint < minMse):
            minMse = msePoint
            minIndex = i
         
    return [minIndex + MIN_ORDER, minMse, mseVec]
    
def GetSmoothTransition (x, y, windowSize, weightsNumber=5):
    ySize = y.size
    result = np.zeros(ySize)
    weights = np.linspace(0.1, 1, weightsNumber) 
    for i in range (0, ySize):
        if i < windowSize:
            result[i] = x[i]
        elif (i>=windowSize) and (i < windowSize + weightsNumber):
            wValue = weights[i-windowSize]
            result[i] = x[i] * (1 - wValue) + y[i] * wValue
        else:
            result[i] = y[i]
    
    return result

def GetResidualExtraction (X, W, T, w0, Q, filterOutput=False):
    decomposition = sm.tsa.seasonal_decompose(X, model='additive')
    X_trend = GetMovingAverage(X.iloc[:,0], W, transitionType='smooth')
    X_sazonal = GetPeriodicMovingAverageOnlyPrediction(decomposition.seasonal.iloc[:,0], T)
    X_residual = X.iloc[:,0] - X_trend - X_sazonal
    b, a = signal.iirnotch(w0, Q)
    if filterOutput:
        return signal.lfilter(b, a, X_residual)
    else:
        return X_residual

#def GetCompleteExtraction(X, W, T, w0, Q):
    

def GetTemporalSeriesCompleteAnalysis(X, PLOT_DIR, title="PLD price",  SAVE_FIGURE=False):
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    MIN_WINDOW_SIZE=3
    bestWLinFit, minMSELinFit, mseLinFit = GetTrendAnalysisByMovingAverageLinFit(X.value, \
                                             title='MSE for trend extraction for ' + str(title) + ' using moving average and linear fit by window size',\
                                             MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                             transitionType='smooth',\
                                             filepath=PLOT_DIR+ str(title) + '_mseTrendLinFitAnalysis.jpg',\
                                             ax=ax1, SAVE_FIGURE=SAVE_FIGURE)
    
    decomposition = sm.tsa.seasonal_decompose(X, model='additive')
    tsaSeasonal = decomposition.seasonal
    ax2 = plt.subplot(2, 1, 2)
    bestTMA,  minTMA, mseTMA = GetSeasonAnalysisByMovingAverageOnly(tsaSeasonal.value, \
                                             title='MSE for seasonal extraction for ' + str(title) + ' using moving average by lag time',\
                                             MIN_WINDOW_SIZE = MIN_WINDOW_SIZE, \
                                             filepath=PLOT_DIR + str(title) + '_mseSeasonalMAOnlyAnalysis.jpg',\
                                             ax=ax2, SAVE_FIGURE = False)

    return bestWLinFit, minMSELinFit, mseLinFit, bestTMA,  minTMA, mseTMA

def EstimateSenoidalCycle(signal, guess_amp=1, guess_freq=0.08, guess_phase=0, guess_mean=0):
    t = np.arange(signal.size)
    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(x[1]*t +x[2]) + x[3] - signal
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    return est_amp*np.sin(est_freq*t+est_phase) + est_mean
    
#OTHERS

##TREND
def ExtractTrendByPolyFit(x, y, originalX, order, FILE_PATH, figureName, showPlot = False, SAVE_FIGURE = False):
    '''
    INPUT:
        x:          array with horizontal axis info
        y:          array with vertical axis info
        orignalX:   array with datetime info aboutX
        order:      order of polynomial used
        FILE_PATH:  folder where file will be saved
        figureName: title shown in figure
        showPlot:   determine whether a plot will be show
        SAVE_FIGURE: enables saving plot feature
    OUTUPUT:
        mse:        Mean Squared Error of polynomial fit
        coef:       Coefficients used to fit data
    '''    
    #linear regression
    coef = pylab.polyfit(x, y, order)
    mse = PlotPolyFit(x, y, originalX, coef, FILE_PATH, figureName, SAVE_FIGURE, showPlot)            
    return [mse, coef]


def GetTrendMSEMovingAverage(y, windowSize, transitionType=None):
    ma = GetMovingAverage(y, windowSize, transitionType=transitionType)
    return mean_squared_error(y, ma)

def GetTrendMSEMovingAverageOnly(y, windowSize, transitionType=None):
    ma = GetMovingAverage(y, windowSize,  PredictFirstWindowPointsMA)
    return mean_squared_error(y, ma)

def GetMovingAverage (y, windowSize, predFunction=PredictFirstWindowPoints, transitionType=None, weightsNumber=5):
    yTemp = predFunction(y, y.size)
    yMA = pd.Series(y).rolling(window=windowSize).mean().iloc[windowSize-1:-1].values
    yMA = np.concatenate( (np.zeros(windowSize), yMA))
    if transitionType == 'smooth':
        return GetSmoothTransition(yTemp, yMA, windowSize, weightsNumber)
    else:
        return np.concatenate((yTemp[0:windowSize], yMA[windowSize:]))


def GetTrendMSEExponentialMovingAverage(y, windowSize):
    ema = GetExponentialMovingAverage(y, windowSize)
    return mean_squared_error(y, ema)


def GetTrendAnalysisByMovingAverageLinFit(y, title, MIN_WINDOW_SIZE,\
                                          transitionType=None, filepath=None, \
                                          ax=None, SAVE_FIGURE=False):
    mseMA = ParameterVariationAnalysis(y, GetTrendMSEMovingAverage, MIN_WINDOW_SIZE, y.size, transitionType=transitionType)
    StemPlotErrorAnalysis(mseMA, MIN_WINDOW_SIZE, xTitle="Window Size",\
                         yTitle="MSE", figTitle=title, filepath=filepath,\
                         ax=ax, SAVE_FIGURE=SAVE_FIGURE)
    return SelectBestParam(mseMA, MIN_WINDOW_SIZE)
    
def GetTrendAnalysisByMovingAverageOnly(y, title, MIN_WINDOW_SIZE, \
                                      ax=None, filepath=None, SAVE_FIGURE=False):

    mseMA = ParameterVariationAnalysis(y, GetTrendMSEMovingAverageOnly, MIN_WINDOW_SIZE, y.size)
    StemPlotErrorAnalysis(mseMA, MIN_WINDOW_SIZE, xTitle="Window Size",\
                         yTitle="MSE", figTitle=title, filepath=filepath,\
                         ax=ax, SAVE_FIGURE=SAVE_FIGURE)
    return SelectBestParam(mseMA, MIN_WINDOW_SIZE)

##TREND

#SEASONAL
def seasonal_mean(x, freq=12):
    """
    Return means for each period in x. freq is an int that gives the
    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
    in the mean.
    """
    nobs = len(x)
    period_averages = np.array([pd_nanmean(x[i::freq], axis=0) for i in range(freq)])
    period_averages -= np.mean(period_averages, axis=0)

    return np.tile(period_averages.T, nobs // freq + 1).T[:nobs]    

def GetExponentialMovingAverage(y, windowSize):
    yTemp = PredictFirstWindowPoints(y, windowSize)
    yEMA = y.ewm(span=windowSize, adjust=False).mean().iloc[windowSize-1:-1].values
    return np.concatenate((yTemp, yEMA))


def GetPeriodicMovingAveragePrediction (y, T, transitionType=None):
    ySize = y.size
    mask = np.zeros(ySize)
    pred = np.zeros(ySize)
    for i in range (0, ySize):
        mask = np.roll(mask, 1)
        if i % T == 0:
            mask[0] = 1
        else:
            mask[0] = 0
        
        idx = np.nonzero(mask)
        vecIdx = y.iloc[idx]
        
        if (i / T) <= 2:
            pred[i] = y[i]
        else:
            fit = np.polyfit(np.arange(0, vecIdx.size - 1), vecIdx[:-1], 1)
            fit_fn = np.poly1d(fit)
            pred[i] = fit_fn(i // T)
        
    return pred

def GetPeriodicMovingAverageOnlyPrediction (y, T):
    ySize = y.size
    mask = np.zeros(ySize)
    pred = np.zeros(ySize)
    for i in range (0, ySize):
        mask = np.roll(mask, 1)
        if i % T == 0:
            mask[0] = 1
        else:
            mask[0] = 0
        
        idx = np.nonzero(mask)
        vecIdx = y.iloc[idx]
        
        if (i / T) <= 2:
            pred[i] = y[i]
        else:
            pred[i] = np.mean(vecIdx[:-1].values)
            
    return pred

def GetMSEfoPeriodicMovingAveragePrediction(y, T, transitionType=None):
    pred = GetPeriodicMovingAveragePrediction(y, T)
    return mean_squared_error (y, pred)

def GetMSEfoPeriodicMovingAverageOnlyPrediction(y, T, transitionType=None):
    pred = GetPeriodicMovingAverageOnlyPrediction(y, T)
    return mean_squared_error (y, pred)

def GetSeasonAnalysisByMovingAverageLinFit(y, title, MIN_WINDOW_SIZE,\
                                          ax=None, filepath=None, SAVE_FIGURE=False):
    mseMA = ParameterVariationAnalysis(y, GetMSEfoPeriodicMovingAveragePrediction, MIN_WINDOW_SIZE, y.size)
    StemPlotErrorAnalysis(mseMA, MIN_WINDOW_SIZE, xTitle="Time Lag",\
                         ax=ax, yTitle="MSE", figTitle=title, filepath=filepath,\
                         SAVE_FIGURE=SAVE_FIGURE)
    return SelectBestParam(mseMA, MIN_WINDOW_SIZE)
    

def GetSeasonAnalysisByMovingAverageOnly(y, title, MIN_WINDOW_SIZE,\
                                          ax=None, filepath=None, SAVE_FIGURE=False):
    mseMA = ParameterVariationAnalysis(y, GetMSEfoPeriodicMovingAverageOnlyPrediction, MIN_WINDOW_SIZE, y.size)
    StemPlotErrorAnalysis(mseMA, MIN_WINDOW_SIZE, xTitle="Time Lag",\
                         ax=ax, yTitle="MSE", figTitle=title, filepath=filepath,\
                         SAVE_FIGURE=SAVE_FIGURE)
    return SelectBestParam(mseMA, MIN_WINDOW_SIZE)

#SEASONAL


#SARIMA
def UseSARIMAToEstimateTemporalSeries(y, PARAM_MAX, INITIAL_TEST_DATE, FILE_PATH,\
                                      param_sarima = None, \
                                      seasonal_param_sarima = None, \
                                      SAVE_FIGURE = False):
    '''
    INPUT:
        y:                  complete data used to fit model
        PARAM_MAX:          max order of parameter used to fit model
        INITIAL_TEST_DATE:  parameter used to separate train from test set
        FILE_PATH:          folder where file will be saved
        param_sarima:       SARIMA param used when no otimization is needed
        param_sarima:       SARIMA seasonal param used when no otimization is needed
        SAVE_FIGURE:        enables saving plot feature
    OUTPUT:
    '''
    #Using ARIMA to extract trend
    p = d = q = range(0, PARAM_MAX)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    yTrain = y[:INITIAL_TEST_DATE]
    yTrain = yTrain[:-1]
    lowestAICparam = []
    lowestAICparamSeasonal = []
    if (param_sarima is None) and (seasonal_param_sarima is None):
        print ('AQUI')
        bestResult = 1000000000
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(yTrain, \
                                                    order=param, \
                                                    seasonal_order=param_seasonal, \
                                                    enforce_stationarity=False, \
                                                    enforce_invertibility=False)
                    
                    results = mod.fit()
                    if results.aic < bestResult:
                        bestResult = results.aic
                        lowestAICparam = param
                        lowestAICparamSeasonal = param_seasonal
                    
                    print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal,\
                          results.aic))
                except:
                    continue
    else:
        lowestAICparam = param_sarima
        lowestAICparamSeasonal = seasonal_param_sarima
     

    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=lowestAICparam,
                                    seasonal_order=lowestAICparamSeasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    figureName = 'DiagnosticBestSARIMA.jpg'
    if SAVE_FIGURE:
        plt.savefig(FILE_PATH + figureName, bbox_inches='tight')
    
    #best result
    ## param: (p, d, q) = (0, 2, 2)
    ## param_seasonal: (p, d, q) = (0, 2, 2, 12)
    pred = results.get_prediction(start=pd.to_datetime(INITIAL_TEST_DATE), dynamic=False)
    pred_ci = pred.conf_int()
    plt.figure()
    ax = y.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    
    
    yPredicted = pred.predicted_mean
    yObserved = y[INITIAL_TEST_DATE:]
    
    mse = mean_squared_error(yObserved, yPredicted)
    ax.set_xlabel('Price')
    ax.set_ylabel('Date')
    plt.legend()
    rmse = np.sqrt(mse)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        r'MSE=%.2f' % (mse, ),
        r'RMSE=%.2f' % (rmse, )
        ))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,\
            verticalalignment='top', bbox=props)
    plt.title('ARIMA ' + str(lowestAICparam)+' S:' + str(lowestAICparamSeasonal) + ' error for test set')
    plt.show()
    figureName = 'BestSARIMA.jpg'
    if SAVE_FIGURE:
        plt.savefig(FILE_PATH + figureName, bbox_inches='tight')

#SARIMA
        
        