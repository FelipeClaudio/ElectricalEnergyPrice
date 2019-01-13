import pandas as pd

def GetRelatedFiles(pDir, initFile):
    file = open(pDir+initFile,'r').readline().rstrip()
    inputTemp = open(pDir+file, 'r').readlines()
    inputFiles = [inp.strip() for inp in inputTemp]
    return inputFiles

class PriceFilesReader:
    _pDir = ""
    _inputFiles = []
    _FILENAMES = {
        'DADGER.RV1' : "NAN",
        'VAZOES.RV1': "VAZOES_RV1.TXT",
        'HIDR.DAT': "NAN",
        'MLT.DAT': "MLT_DAT.TXT",
        'PERDAS.DAT': "NAN",
        'DADGNL.RV1': "NAN"
    }
    _POSTOSFILE = 'POSTOS_MODIFICADO_DAT.TXT'
    _VAZOESDATFILE = 'VAZOES_DAT.TXT'
    _PRECOSFILE = 'precos.csv'
    _HIDRFILE = 'HIDR.csv'

    _months = ['Janeiro', 'Fevereiro', 'Marco', 'Abril', 'Maio',
            'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro',
            'Novembro', 'Dezembro']

    #get working directory and input files from initial file
    def __init__ (self, pDir, initFile):
        self._pDir = pDir
        self._inputFiles = GetRelatedFiles(self._pDir, initFile)
        self.GetInputs()

    def GetInputs (self):
        for inp in self._inputFiles:
            self.SelectInputType(inp)
        print 

    def SelectInputType (self, inputName):
        #check if input type is mapped in class and select the right function to read the file
        if (self._FILENAMES.get(inputName, None) is not None):
            keys = list(self._FILENAMES.keys())
            id = keys.index(inputName)
            func = self._functions[id]
            if (func is not None):
                func(self, inputName)
            else:
                print("Function need to read file " + inputName + ' is not yet implemented')
            

    def ReadMLT(self, inputName):
        data = pd.read_csv(self._pDir + self._FILENAMES[inputName], sep='\s+',
        names = ['Posto'] + self._months)
        self._mlt = data
    
    def ReadVazoes(self, inputName):
        data = pd.read_csv(self._pDir + self._FILENAMES[inputName], sep='\s+',
        names = ['Posto', 'Ano'] + self._months)
        self._vazoes = data

    ##Conversar depois sobre necessidade de ler os postos todas as vezes - REMOVER ESTE COMENTÁRIO
    def ReadPostos (self):
        data = pd.read_csv(self._pDir + self._POSTOSFILE , index_col=False,
        names = ['Id Posto', 'Nome', 'Ano Inicio', 'Ano Fim'])
        self._postos = data
        
    def ReadVazoesDAT (self):
        data = pd.read_csv(self._pDir + self._VAZOESDATFILE, sep='\s+',
        names = ['Posto', 'Ano'] + self._months)
        self._vazoesDAT = data
        
    def ReadPrecos (self, mainDir):
        data = pd.read_csv(mainDir + '/' + self._PRECOSFILE, sep=';',
        parse_dates = True)
        self._precos = data
    
    def ReadHidr (self):
        data = pd.read_csv(self._pDir + self._HIDRFILE, parse_dates = True, sep=";")
        self._hidr = data

    def GetPostos(self):
        self.ReadPostos()
        return self._postos

    def GetMLT(self):
        return self._mlt

    def GetVazoes(self):
        return self._vazoes
    
    def GetVazoesDAT(self):
        self.ReadVazoesDAT()
        return self._vazoesDAT
    
    def GetPrecos(self, mainDir):
        self.ReadPrecos(mainDir)
        return self._precos
    
    def GetHidr(self):
        self.ReadHidr()
        return self._hidr
        
    #Reading function for each file
    _functions = [None, ReadVazoes, None, ReadMLT, None, None]