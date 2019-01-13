# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:36 2018

@author: felipe
"""
import InputReader as inp
import os

#read main file
currDir = os.getcwd()
fileDir = os.path.abspath(os.path.join(currDir, os.pardir))
mainDir = fileDir
fileDir = fileDir + '/PLD_Outubro_2018/'
initialDirPrices = fileDir + '/10_out18_RV0_logENA_Mer_d_preco_m_0/'
initialFile = 'CASO.DAT'

print (fileDir)
print (initialDirPrices)
pr = inp.PriceFilesReader(initialDirPrices, initialFile)


mlt = pr.GetMLT()
#print (mlt)
vazoes = pr.GetVazoes()
#print(vazoes)
postos = pr.GetPostos()
#print (postos)
##Posto com nome problematico -> Tiete
#print(postos.iloc[107])
vazoesDAT = pr.GetVazoesDAT()
#print(vazoesDAT)
precos = pr.GetPrecos(mainDir)
#print(precos)
hidr = pr.GetHidr()
#print(hidr)

inputs = {
        'vazoes': vazoes,
        'postos': postos,
        'vazoesDAT': vazoesDAT,
        'hidr': hidr  
}

output = {
        'precos': precos
}