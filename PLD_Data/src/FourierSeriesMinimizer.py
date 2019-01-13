#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 19:50:10 2019

@author: felipe
"""
import numpy as np
from scipy.optimize import minimize

class FourierSeriesMinimizer:
    def SetTimeVector (self, timeSerie):
        self._tms = timeSerie
    
    def GetTimeVector(self):
        return self._tms
    
    def SetRefenceValue(self, referenceValue):
        self._refVal = referenceValue
        
    def GetReferenceValue(self):
        return self._refVal
    
    def TimeComponentDifference(self, params):
        a, b, f = params
        return np.sum(np.square(self._refVal - a * np.cos( 2 * np.pi * f * self._tms) + \
                                       b * np.sin( 2 * np.pi * f * self._tms)))
        
    def MinimizeFourierSeriesCoefficients(self, initial_guess):
        result = minimize(self.TimeComponentDifference, initial_guess, method='Nelder-Mead')
        if result.success:
            fitted_params = result.x
            return [fitted_params]
        else:
            raise ValueError(result.message)
            
    def ExtractSenoidalInformation(self, an, bn, fn, originalSerie):
        yn = an * np.cos( 2 * np.pi * fn * self._tms) + bn * np.sin( 2 * np.pi * fn * self._tms)
        return originalSerie - yn