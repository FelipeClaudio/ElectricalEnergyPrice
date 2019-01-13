# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:16:48 2018

@author: felip
"""
import numpy as np
import calendar
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

class DataPlotter:
    def __init__ (self, path):
        self._MONTHS = calendar.month_name[1:13]
        self._path = path
        
    def plotVazoes(self, postos, years, val, plotName):
        plt.figure()
#        formattedYears = self.getXAxisLabels(years)
        startDate = date.toordinal(date(years[0],1,1))
        endDate = date.toordinal(date(years[-1],1,1))
        dateVec = np.linspace(startDate, endDate, (startDate - endDate))
        for posto in postos:
            inputDataTemp = val.loc[val['Posto'] == posto]
            inputDataTemp = inputDataTemp.loc[inputDataTemp["Ano"] >= years[0]]
            inputDataTemp = inputDataTemp.loc[inputDataTemp["Ano"] <= years[-1]]
            inputData = inputDataTemp.iloc [:,2:]
            inputList = self.dataToList(inputData)
            inputSeries = pd.Series(inputList)
            plt.xticks(np.arange(inputSeries.size), formattedYears , rotation=90)
            inputSeries.plot(use_index = False, marker = 'x', label="Posto " + str(posto))
            
        plt.legend(loc="best")
        plt.title(plotName)            
        plt.show()
        plt.savefig(self._path + plotName + ".jpg", bbox_inches='tight')
                
            

    def dataToList(self, val):
        temp=[]
        for row in val.iterrows():
            index, data = row
            temp = temp + data.tolist() 
        return temp

    def getXAxisLabels(self, years):
        dateList = []
        for y in years:
            for m in self._MONTHS:
                dateTemp = str(y) + '-' + m
                dateList.append(dateTemp)
        
        return dateList
        
    def setPath (self, path):
        self._path = path