import pandas as pd

def GetRelatedFiles(pDir, initFile):
    file = open(pDir+initFile,'r').readline().rstrip()
    inputTemp = open(pDir+file, 'r').readlines()
    inputFiles = [inp.strip()[30::] for inp in inputTemp]
    return inputFiles

class GenericInputReader:
    _mainDir = ""
    _inputFiles = []
    _dictDF = {}

    #Get working directory and input files from initial file
    def __init__ (self, mainDir, initFile):
        self._mainDir = mainDir
        self._inputFiles = GetRelatedFiles(self._mainDir, initFile)
        self.GetDataFromFiles()
     
    #Read all files listed in main file which have default format 
    def GetDataFromles (self):
        for inp in self._inputFiles:
            try:
                data = pd.read_csv(self._mainDir +inp, parse_dates = True, skiprows=[1], sep="\s+")
            except:
                data = None
                
            self._dictDF.update({inp[:-4] : data})
            
    def GetAllData (self):
        return self._dictDF
        