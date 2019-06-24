#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:41:59 2019

@author: felipe
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pylab import rcParams
import statsmodels.api as sm
import locale
import seaborn as sns
import math
import utilities as util
from scipy import signal
from pandas.plotting import autocorrelation_plot

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
plt.close('all')

mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")

'''
fig = plt.figure()
ax = fig.gca()
minTemp = pd.read_csv('mean-daily-temperature-fisher-ri.csv', parse_dates=['Date'], \
                           date_parser=mydateparser)

minTemp.set_index('Date', drop=True, inplace=True)
minTemp.columns = ['Temperatura']
#minTemp.plot(ax=ax)
plt.xlabel('Data (dias)')
plt.ylabel('Temperatura ($^\circ$C)')

decomposition = sm.tsa.seasonal_decompose(minTemp, model="additive")
seasonal = decomposition.seasonal

plt.subplot(2, 1, 1)
plt.plot(minTemp.index, minTemp.Temperatura)
plt.subplot(2, 1, 2)
plt.plot(seasonal, label='Sazonalidade')
plt.show()
#decomposition.plot()
'''

'''
mydateparser2 = lambda x: pd.datetime.strptime(x, "%y-%m")
sShamp = pd.read_csv('sales-of-shampoo-over-a-three-ye.csv', parse_dates=['Month'], \
                           date_parser=mydateparser2)
sShamp.set_index('Month', drop=True, inplace=True)
sShamp.columns = ['Vendas']
#sShamp.plot(ax=ax)

decomposition = sm.tsa.seasonal_decompose(sShamp, model="additive", extrapolate_trend=True)
decomposition.plot()

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = plt.figure()
ax = fig.gca()

plt.plot(sShamp.index, sShamp.Vendas)
plt.plot(trend, label='Tendência')
plt.plot(seasonal, label='Sazonalidade')
plt.plot(residual, label='Resíduo')
plt.title('Venda mensal de shampoo')
plt.xlabel('Mês')
plt.ylabel('Vendas de shampoo')
plt.legend()
plt.show()
'''

mydateparser3 = lambda x: pd.datetime.strptime(x, "%Y-%m")
bornNy = pd.read_csv('monthly-new-york-city-births-unk.csv', parse_dates=['Mes'], \
                           date_parser=mydateparser3, index_col=['Mes'])


decomposition = sm.tsa.seasonal_decompose(bornNy, model="additive", extrapolate_trend=True)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

bornNy.plot(title='Nascimentos em Nova York entre 1946-1959')
plt.figure()
plt.plot(bornNy.index, bornNy.Nascimentos)
plt.plot(trend, label='Tendência')
plt.xlabel('Mês de Nascimento')
plt.ylabel('Nascimentos')
plt.legend()
plt.title('Nascimento em Nova York e Tendência')
plt.show()

bornNy.plot(title='Nascimentos em Nova York entre 1946-1959')
plt.figure()
plt.plot(bornNy.index, bornNy.Nascimentos)
plt.plot(seasonal, label='Sazonalidade')
plt.xlabel('Mês de Nascimento')
plt.ylabel('Nascimentos')
plt.legend()
plt.title('Nascimento em Nova York e Sazonalidade')
plt.show()

normRes = util.NormalizeSeries(residual)
plt.figure()
ax1 = plt.subplot(1, 2, 1)
util.FFT(normRes, 'Frequência', 'Magnitude',  title='FFT do sinal residual normalizado', showPlot=True, ax=ax1)
ax2 = plt.subplot(1, 2, 2)
util.PlotDistribution(normRes, 'Magnitude', 'Número de Ocorrências', \
                      plotTitle='Distribuição do sinal residual normalizado', ax=ax2)

w0 = 0.135  # Frequency to be removed from signal (Hz)
Q = 0.2  # Quality factor
# Design notch filter
b, a = signal.iirnotch(w0, Q)
filteredResidual = signal.lfilter(b, a, normRes)
ax1 = plt.figure()
plt.subplot(1, 2, 1)
util.FFT(filteredResidual, 'Frequência', 'Magnitude', title='FFT do sinal residual normalizado pós filtragem', showPlot=True, ax=ax1)
ax2= plt.subplot(1, 2, 2)
util.PlotDistribution(filteredResidual, 'Magnitude', 'Número de Ocorrências', \
                      plotTitle='Distribuição do sinal residual normalizado pós filtragem', ax=ax2)

plt.figure()
plt.plot(bornNy.index, filteredResidual)
plt.title('Sinal Residual Normalizado Pós Filtragem')
plt.xlabel('Mẽs')
plt.ylabel('Amplitude')

sigma = np.std(filteredResidual)
ax = autocorrelation_plot(normRes)
ax.set_title('Autocorrelação do sinal residual')

ax = autocorrelation_plot(filteredResidual)
ax.set_title('Autocorrelação do sinal residual filtrado')