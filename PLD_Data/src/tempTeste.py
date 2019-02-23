#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 06:22:20 2019

@author: felipe
"""

import trendAnalysis as tr
import pandas as pd
import numpy as np
import utilities as util


def GetPeriodicMovingAveragePrediction (y, T):
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


    


def GetPeriodicMovingAveragePredictionWithWindow (y, T, windowSize = 2):
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
            vIdxSize = vecIdx.size
            fit = np.polyfit(np.arange(0, windowSize), vecIdx[(vIdxSize - 1 -windowSize):-1], 1)
            fit_fn = np.poly1d(fit)
            pred[i] = fit_fn(vIdxSize  - 1 -windowSize)
            
    return pred

'''
ROOT_FOLDER  = '/home/felipe/Materias/TCC/'
ONS_DIR = ROOT_FOLDER + 'PLD_Data/ONS_DATA'
t = util.ReadONSEditedCSV( ONS_DIR + '/Simples_Energia_Armazenada_Mês_data_editado.csv', 'Total Stored Energy')
a = util.ExtractTrainTestSetFromTemporalSeries(t, '01/2015', '12/2018', '01/2018')[1]
c = util.ExtractTrainTestSetFromTemporalSeries(t, '01/2015', '12/2016', '12/2015')

T = 3
vetorTeste = np.arange(10) + 1
vetorTeste = np.append(vetorTeste, [8.5, 6.4, 5.3, 8.7, 4.9])
pandasSeries = pd.Series(vetorTeste)
resp = GetPeriodicMovingAveragePrediction(pandasSeries, 3)

from sklearn.preprocessing import normalize
a = np.array([1, 2, 5])
b = normalize(a.reshape(-1, 1), norm='max', axis=0)
b = np.array(b.T)[0]

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math

plt.close('all')
w0 = 0.08  # Frequency to be removed from signal (Hz)
Q = 0.1  # Quality factor
#Q = math.sqrt(2)
# Design notch filter
b, a = signal.iirnotch(w0, Q)
t = pldResidue.values
t = util.NormalizeSeries(t.reshape(-1,1))
util.FFT(t, 'a', 'b', 'c1', 'd', showPlot=True)
resp = signal.lfilter(b, a, t)
util.PlotDistribution(resp, 'ax1', 'yx1', 'px1', 'df')
util.FFT(resp, 'a', 'b', 'c2', 'd', showPlot=True)


w0 = 0.32  # Frequency to be removed from signal (Hz)
Q = 0.1  # Quality factor
# Design notch filter
b, a = signal.iirnotch(w0, Q)
resp = signal.lfilter(b, a, t)
util.PlotDistribution(resp, 'ax2', 'yx2', 'px2', 'df')
util.FFT(resp, 'a', 'b', 'c3', 'd', showPlot=True)

import matplotlib.pyplot as plt

y = np.arange(20) + 1
ySize = y.size
windowTest = ySize
windowSize = 3
weightsNumber = 10
yFinal = tr.GetSmoothMovingAverage(y, windowSize, weightsNumber=10)
yTemp = tr.PredictFirstWindowPoints(y, windowTest)
yMA = pd.Series(y).rolling(window=windowSize).mean().iloc[windowSize-1:-1].values
yMA = np.concatenate( (np.zeros(windowSize), yMA) )
weights = np.linspace(0.1, 1, weightsNumber)

yFinal = [0]*y.size
for i in range (0, ySize):
    if i < windowSize:
        yFinal[i] = yTemp[i]
    elif (i>=windowSize) and (i < windowSize + weightsNumber):
        wValue = weights[i-windowSize]
        yFinal[i] = yTemp[i] * (1 - wValue) + yMA[i] * wValue
    else:
        yFinal[i] = yMA[i]

plt.figure()
yOld = tr.GetMovingAverage(y, windowSize)
plt.subplot(2,1,1)
plt.plot(yOld)
plt.subplot(2,1,2)
plt.plot(yFinal)
y[10:]
'''
val = index=np.arange(10)
mask = np.array([1, 4, 5])
a = pd.DataFrame(index = val, columns = ['Valor'])
a['Valor'] = val
a.index.name = 'idx'
b = util.GetFilteredSeriesByMask(a, mask=mask, shiftLeft=0)[1]
