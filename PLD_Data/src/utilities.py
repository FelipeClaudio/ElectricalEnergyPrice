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

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

def FFT(y, xlabel, ylabel, title, figureName, T=1.0, \
        axText="0.5rad/s = 2 months", \
        showPlot=False, saveFig=False):
    # Number of sample points
    N = y.size
    # sample spacing
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    
    if showPlot:
        plt.figure()
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        ax = plt.gca()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, axText, transform=ax.transAxes, fontsize=14,\
                verticalalignment='top', bbox=props)
        plt.grid()
        plt.show()
        
        if saveFig:
            plt.savefig(figureName, bbox_inches='tight')  
