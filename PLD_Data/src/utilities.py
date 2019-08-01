#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 20:02:50 2018

@author: felipe
"""
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pylab import rcParams
from datetime import datetime, timedelta
from collections import OrderedDict
import statsmodels.api as sm 
from sklearn.preprocessing import normalize
import math
import seaborn as sns
import pandas as pd
import copy
from enum import Enum

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

class language(Enum):
    US=0
    PT=1

#defaultMask = np.array([2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20])
#older to newer - REVERSERD ORDER
#ex:
#a = [1, 2, 3, 17, 18, 19 ,20]
#b = [19, 18, 17, 3, 2, 1, 0]
#c = [0, 1, 2, 3, 17, 18, 19]
defaultMask = np.array([0, 1, 2, 3, 4, 5, 6, 18, 19])

def FFT(y, xlabel, ylabel, title, figureName, T=1.0, \
        axText="0.5rad/s = 2 meses", \
        ax=None, showPlot=False, SAVE_FIGURE=False):
    # Number of sample points
    N = y.size
    # sample spacing
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    #print(xf)
    #print(yf)
    
    if showPlot:
        if ax is None:
            plt.figure()
            plt.title(title)
        plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        ax = plt.gca()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, axText, transform=ax.transAxes, fontsize=14,\
                verticalalignment='top', bbox=props)
        plt.grid()
        plt.show()
        
        if SAVE_FIGURE:
            plt.savefig(figureName, bbox_inches='tight')  

def PlotDistribution(y, xTitle, yTitle, plotTitle, filepath, ax=None, SAVE_FIGURE=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(plotTitle)
    sns.distplot(y, bins=math.ceil(4*math.sqrt(y.size)), ax=ax)  
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.show()
    if SAVE_FIGURE:
        plt.savefig(filepath, bbox_inches='tight')
    
    

def GetMonthRange(initialDate, finalDate, dateFormat='%Y-%m-%d'):
    dates = [initialDate, finalDate]
    start, end = [datetime.strptime(_, dateFormat) for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1, 1))
    return mlist


def NormalizeSeries(y): 
    yNormalized = normalize(y, norm='max', axis=0)
    return np.array(yNormalized.T)[0]

def PlotTSA(original, trend, seasonal, residual, \
            originalPlotTitle, originalPlotFileName, \
            resultPlotTitle, resultPlotFileName, SAVE_FIGURE=False):

    decomposition = sm.tsa.seasonal_decompose(original, model='additive')
    decomposition.plot()
    plt.suptitle(originalPlotTitle)

    if SAVE_FIGURE:
        plt.savefig(originalPlotFileName, bbox_inches='tight')
    
    plt.figure()
    plt.suptitle(resultPlotTitle)
    
    plt.subplot(4, 1, 1)
    plt.plot(original)
    plt.ylabel('Original')
    
    plt.subplot(4, 1, 2)
    plt.plot(original.index, trend)
    plt.ylabel('Trend')
    
    plt.subplot(4, 1, 3)
    plt.plot(original.index, seasonal)
    plt.ylabel('Seasonal')
    
    plt.subplot(4, 1, 4)
    plt.plot(original.index, residual)
    plt.ylabel('Residual')
    
    if SAVE_FIGURE:
        plt.savefig(resultPlotFileName, bbox_inches='tight')
        
def ExtractTrainTestSetFromTemporalSeries (data, initialDate = None, \
                                           finalDate = None, timeSplit = None):
    if initialDate is None:
        initialDate = data.index[0]
    
    if finalDate is None:
        finalDate = data.index[-1]
        
    if timeSplit is None:
        timeSplit = finalDate
        
    data = data.loc[data.index >= initialDate]
    data = data.loc[data.index <= finalDate]
    dataTrain = data.loc[data.index <= timeSplit]
    dataTest = data.loc[data.index > timeSplit]
    return [dataTrain, dataTest]

def ReadONSEditedCSV(filename, columnName, dateParser=None, INITIAL_DATE = None, \
                     FINAL_DATE = None, TIME_SPLIT = None):
    if dateParser is None:
        mydateparser = lambda x: pd.datetime.strptime(x, "%B %Y")
    else:
        mydateparser = dateParser   
    data = pd.read_csv(filename,\
                           parse_dates=['month'], \
                           date_parser=mydateparser)
    data.set_index('month', inplace=True)
    data.columns = [columnName]
    data = data.sort_index()
    return ExtractTrainTestSetFromTemporalSeries(data, initialDate=INITIAL_DATE,\
                                                 finalDate = FINAL_DATE, \
                                                 timeSplit = TIME_SPLIT)
       
def GetFilteredSeriesByMask(data, shiftLeft=0, mask=None):
    #get indexed data to perform operations
    filteredSeries = data.reset_index()
    if mask is None:
        #Need to deep copy in order to no alter global variable in every interaction
        mask = copy.deepcopy(defaultMask)
        
    mask += shiftLeft
    filteredSeries = filteredSeries.iloc[mask] 
    indexColumn = data.index.name
    filteredSeries.set_index(indexColumn, inplace=True)
    filteredSeries = filteredSeries.sort_index()
    '''
    if shiftLeft == 0:
        latestValue = data[-1:]
    else:
        latestValue = data[(-shiftLeft-1):-shiftLeft]
    '''
    #The latest elemnt is um after the lats element of mask
    latestValue = data.iloc[[np.max(mask) + 1]]
    latestValue = latestValue.reset_index()
    latestValue.set_index(indexColumn, inplace=True)
    
    return [latestValue, filteredSeries.sort_index(ascending=False)]

def TransposeAndSetColumnNames(data, prefix, mask=None):
    if mask is None:
        mask = copy.deepcopy(defaultMask)
    
    mask = np.fliplr([mask])[0]
    mask -= (np.max(mask) + 1)
    mask = np.abs(mask)
    columnNames = [''] * mask.size
    for col in range (0, mask.size):
        columnNames[col] = prefix + str(mask[col])
    
    data = data.transpose()
    data.columns = columnNames
    return data.reset_index(drop=True)

def GetDefaultMask():
    return defaultMask
