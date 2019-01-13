#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan c 6 18:25:56 2019

@author: felipe
"""
import pandas as pd
import numpy as np

from scipy.optimize import minimize

'''
def timeComponent(a, b, f, s):
    squareValue = np.square(s - a * np.cos( 2 * np.pi * f) +b * np.sin( 2 * np.pi * f))
    sumValue = np.sum(squareValue)
    normalization = (1 / squareValue.size) * sumValue
    
    return normalization
'''

def timeComponent(params):
    # print(params)  # <-- you'll see that params is a NumPy array
    s = np.array([2, -2, 2])
    t = np.linspace(0, 1, 3)
    a, b, f = params # <-- for readability you may wish to assign names to the component variables
    return np.sum(np.square(s - a * np.cos( 2 * np.pi * f * t) +b * np.sin( 2 * np.pi * f * t)))


initial_guess = [1.6, 0.4, 2.9]

result = minimize(timeComponent, initial_guess)
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)

an = result.x[0]
bn = result.x[1]
fn = result.x[2]



'''
timeComponent([0, 5, 7])
params = [2, 0, 1]
a, b, f = params # <-- for readability you may wish to assign names to the component variables
print('AQUI')
print(s - a * np.cos( 2 * np.pi * f * t) + b * np.sin( 2 * np.pi * f * t))

'''
t = np.linspace(0, 1, 3)
yn = an * np.cos( 2 * np.pi * fn * t) + bn * np.sin( 2 * np.pi * fn * t)