# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:19:29 2018

@author: felip
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:36 2018

@author: felipe
"""
import GenericInputReader as inp
import os

#read main file
currDir = os.getcwd()
fileDir = os.path.abspath(os.path.join(currDir, os.pardir))
mainDir = fileDir
fileDir = fileDir + '/NEWAVE_DATA_INPUT/'
initialDirPrices = fileDir #+ '/10_out18_RV0_logENA_Mer_d_preco_m_0/'
initialFile = 'CASO.DAT'

print (fileDir)
print (initialDirPrices)
gi = inp.GenericInputReader(initialDirPrices, initialFile)

data = gi.GetAllData()