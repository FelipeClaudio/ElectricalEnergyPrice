# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:36 2018

@author: felipe
"""
import InputReader as inp
import numpy as np
import os
import PlotData as plotData
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date


#read main file
currDir = os.getcwd()
fileDir = os.path.abspath(os.path.join(currDir, os.pardir))
mainDir = fileDir
fileDir = fileDir + '/DECOMP_DATA_INPUT/'
initialDirPrices = fileDir + 'Patamar_Sombra_sem2/'
initialFile = 'CASO.DAT'

print (fileDir)
print (initialDirPrices)
pr = inp.PriceFilesReader(initialDirPrices, initialFile)
pr.ReadPrecos(mainDir)

inputs = {
        'vazoes': pr.GetVazoes(),
        'postos': pr.GetPostos(),
        'vazoesDAT': pr.GetVazoesDAT(),
        'hidr': pr.GetHidr()  
}

output = {
        'precos': pr.GetPrecos(mainDir),
        'pldMensal': pr.GetPLDMedio("C:/Users/felip/Documents/Materias/TCC/PLD_Data/PLD_medio.csv")
}

hidr = inputs["hidr"]
vazoes = inputs["vazoes"].fillna(0)

yearsRV1 = np.arange(1962, 1971)

pData = plotData.DataPlotter("C:/Users/felip/Documents/Materias/TCC/PLD_Data/src/plots/VazoesDAT/")
vazoesDAT = inputs["vazoesDAT"]

yearsDAT = np.arange(2010, 2019)

POSTOSSUDESTE = []

POSTOSSUDESTE.append([1, 2, 211, 6, 7, 8, 9, 10, 11, 12])
POSTOSSUDESTE.append([14, 15, 16, 17, 18, 22, 251, 20, 24])
POSTOSSUDESTE.append([25, 206, 207, 28, 205, 209, 31, 32, 33])
POSTOSSUDESTE.append([34, 237, 238, 239, 240, 242, 243, 244, 245]) 
#                 246, 47, 48, 49, 50, 51, 52, 61, 62, 63, 266, 
#                 160, 161, 104, 109, 117, 118, 116, 120, 121, 122, 
#                 123, 202, 125, 197, 198, 129, 300, 130, 300, 202, 202, 
#                 134, 263, 267, 149, 141, 148, 144, 255, 258, 154, 155, 
#                 156, 158, 201, 203, 300, 202, 300, 135, 199, 262, 183, 
#                 265, 295, 296, 23, 196, 225, 227, 228, 229, 230, 241, 
#                 249, 270, 191, 253, 257, 273, 294, 187, 145, 278, 279, 
#                 7281, 282, 283, 285, 287, 289, 99, 58, 259, 252, 291, 
#                 247, 248, 313, 261, 55, 54, 57]

#idx = 1
#for postos in POSTOSSUDESTE:
#    pData.plotVazoes(postos, yearsDAT, vazoesDAT, 
#                     "VazoesDAT" + str(idx) + " " + str(yearsDAT[0]) + "-" + str(yearsDAT[-1]))
#    idx = idx + 1
#    
#idx = 1
#for postos in POSTOSSUDESTE:
#    pData.plotVazoes(postos, yearsRV1, vazoesDAT, 
#                     "VazoesDAT" + str(idx) + " " + str(yearsRV1[0]) + "-" + str(yearsRV1[-1]))
#    idx = idx + 1    
#
#idx = 1
#pData.setPath("C:/Users/felip/Documents/Materias/TCC/PLD_Data/src/plots/VazoesRV1/")
#for postos in POSTOSSUDESTE:
#    pData.plotVazoes(postos, yearsRV1, vazoesDAT, 
#                     "VazoesRV1_" + str(idx) + " " + str(yearsRV1[0]) + "-" + str(yearsRV1[-1]))
#    idx = idx + 1
   
##transformar em metodo

#plt.figure()
#pldInfo = output["precos"]
#pldSE = pldInfo[["Data Início", "Pesado SE", "Médio SE", "Leve SE"]]
##pldSE.plot(x="Data Início",  y="Pesado SE")
#plt.plot(np.arange(909), pldSE["Pesado SE"])
#plt.plot(np.arange(909), pldSE["Médio SE"])
#plt.plot(np.arange(909), pldSE["Leve SE"])
#plt.xticks(np.arange(909), pldSE["Data Início"] , rotation=90)
#plt.legend(loc="best")
#plt.show()

#pldMedio =  pd.read_csv("C:/Users/felip/Documents/Materias/TCC/PLD_Data/PLD_medio.csv", sep="\s+", parse_dates=True, header=None)
#pldMedio = pldMedio.sort_index(axis=0, ascending=False)
#plt.plot(np.arange(1, pldMedio[0].size + 1), pldMedio[1])
#plt.xticks(np.arange(pldMedio[0].size), pldMedio[0] , rotation=90)
#plt.axis(x="Data", y="Valor")
#plt.legend(loc="best")
#plt.title("PLD MédIO - 05/2003 - 09/2018")
#plt.show()

#corr1 = pldMedio[1]
#corr[:,1] = 

#startDate = date.toordinal(date(2003,5,31))
#endDate = date.toordinal(date(2018,9,30))
#t = np.linspace(startDate,endDate, 185)
#datas = []
#for t2 in t:
#    datas.append(date.fromordinal(int(t2)))
#
#ax = plt.gca()
#ax.XTick = datas
#plt.plot_date(datas,pldMedio[1], xdate=True)
#
#print ((startDate - endDate)/30)
#print (12 * 8)