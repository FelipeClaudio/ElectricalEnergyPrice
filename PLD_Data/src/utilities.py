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

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

def FFT(y, xlabel, ylabel, title, figureName, T=1.0, \
        axText="0.5rad/s = 2 months", \
        showPlot=False, SAVE_FIGURE=False):
    # Number of sample points
    N = y.size
    # sample spacing
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    if showPlot:
        plt.figure()
        plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        ax = plt.gca()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, axText, transform=ax.transAxes, fontsize=14,\
                verticalalignment='top', bbox=props)
        plt.grid()
        plt.show()
        
        if SAVE_FIGURE:
            plt.savefig(figureName, bbox_inches='tight')  

def PlotDistribution(y, xTitle, yTitle, plotTitle, filepath, SAVE_FIGURE=False):
    fig = plt.figure()
    ax = fig.gca()
    sns.distplot(y, bins=math.ceil(math.sqrt(y.size)), ax=ax)
    ax.set_title(plotTitle)
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.show()
    if SAVE_FIGURE:
        plt.savefig(filepath, bbox_inches='tight')

def GetMonthRange(initialDate, finalDate):
    dates = [initialDate, finalDate]
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1, 1))
    return mlist


def NormalizeSeries(y): 
    yNormalized = normalize(y, norm='max', axis=0)
    return np.array(yNormalized.T)[0]

def PlotTSA(original, trend, seasonal, \
            originalPlotTitle, originalPlotFileName, \
            resultPlotTitle, resultPlotFileName, SAVE_FIGURE=False):

    decomposition = sm.tsa.seasonal_decompose(original, model='additive')
    decomposition.plot()

    if SAVE_FIGURE:
        plt.savefig(originalPlotFileName, bbox_inches='tight')
    
    residual = original.values - trend - seasonal
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
    plt.plot(original.index, NormalizeSeries(residual))
    plt.ylabel('Residual')
    
    if SAVE_FIGURE:
        plt.savefig(resultPlotFileName, bbox_inches='tight')    
