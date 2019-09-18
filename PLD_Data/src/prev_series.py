#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:16:27 2019

@author: felipe
"""
import numpy as np
import trendAnalysis as tr



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




windowSize = 3
n_steps = 1

x = np.arange(10) + 1

y = predict_deterministc_series(x, n_steps, windowSize)
y2 = tr.GetMovingAverage(x, windowSize)
