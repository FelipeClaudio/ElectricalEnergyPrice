import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pylab import rcParams
import statsmodels.api as sm
import trendAnalysis as tr
import utilities as util
from FourierSeriesMinimizer import FourierSeriesMinimizer
import utilities as util

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

rcParams['figure.figsize'] = 18, 8

#setting plotting window parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

ROOT_FOLDER  = '/home/felipe/Materias/TCC/'
#loading PLD data
MAIN_DIR = ROOT_FOLDER + '/PLD_Data/PLD_Outubro_2018'
MAIN_DIR += '/10_out18_RV0_logENA_Mer_d_preco_m_0/'
mydateparser = lambda x: pd.datetime.strptime(x, "%m/%Y")
meanPLD = pd.read_csv(MAIN_DIR + 'PLD_medio.csv', \
                      parse_dates=['Mês'], sep="\\s+",\
                      date_parser=mydateparser)
mPLDSE = meanPLD[['Mês', 'SE/CO']]
mPLDSE.columns = ['month', 'price']
mPLDSE.set_index('month', inplace=True)
mPLDSE = mPLDSE.sort_index()

#spliting into training/validation and test set
TRAIN_STARTING_DATE = '2015-01-01'
TEST_STARTING_DATE = '2018-01-01'
mPLDSE = mPLDSE[mPLDSE.index >= TRAIN_STARTING_DATE]
trainValidationSet = mPLDSE.loc[mPLDSE.index < TEST_STARTING_DATE]
testSet = mPLDSE.loc[mPLDSE.index >= TEST_STARTING_DATE]

#Get train/validation X test set ratio
trainTestRatio = (trainValidationSet.size / (testSet.size + trainValidationSet.size)) * 100
print(trainTestRatio)

#reset index and separate dataset into x and y values
trSet = trainValidationSet
trSet = trSet.reset_index(drop='true')
xTrain = np.array(trSet.index)
yTrain = np.array(trSet.price)

testSetTemp = testSet
testSetTemp = testSetTemp.reset_index(drop='true')
xTest = np.array(testSetTemp.index)
yTest = np.array(testSetTemp.price)

PLOT_DIR = ROOT_FOLDER + '/PLD_Data/src/plots/SeriesTemporais/'

SAVE_PLOT = False
SHOW_PLOT = False

#using train set to determine coefficients
#and using this coeficients to predict future trends
INITIAL_ORDER = 1
FINAL_ORDER = 11
mseTraining = np.zeros(FINAL_ORDER - INITIAL_ORDER)
mseTest = np.zeros(FINAL_ORDER - INITIAL_ORDER)
for order in range(INITIAL_ORDER, FINAL_ORDER):
    figureName = 'Poly' + str(order) + 'Order.jpg'
    testFigureName = 'Test' + str(order) + 'Order.jpg'
    imageName = 'Trend extraction using ' + str(order)\
    + 'º order polynomial regression'
    testImageName = 'Error in test set using ' + str(order)\
    + 'º order polynomial regression'

    #plot train set trend
    mse, coef = tr.ExtractTrendByPolyFit(xTrain, yTrain, trainValidationSet.index,\
                                   order, PLOT_DIR + figureName,\
                                   imageName, SHOW_PLOT, SAVE_PLOT)
    mseTraining[order - INITIAL_ORDER] = mse[0]

    #plot test set trend
    mse = tr.PlotPolyFit(xTest, yTest, testSet.index, coef,\
                   PLOT_DIR + testFigureName,\
                   testImageName, SHOW_PLOT, SAVE_PLOT)
    mseTest[order - INITIAL_ORDER] = mse[0]


#plotting error found in trains and tests by order
SAVE_PLOT = False

figureName = 'MSEXOrder_training.jpg'
imageName = 'MSE in training set by polinomial order'
tr.PlotErrorFunction(xTrain, mseTraining, INITIAL_ORDER, FINAL_ORDER, \
                      PLOT_DIR + figureName, imageName, yLabel='RMSE', \
                      xLabel='Degree', SAVE_FIGURE=SAVE_PLOT)

figureName = 'RMSEXOrder_training.jpg'
imageName = 'RMSE in training set by polinomial order'
rmseTrainning = np.sqrt(mseTraining)
tr.PlotErrorFunction(xTrain, rmseTrainning, INITIAL_ORDER, FINAL_ORDER, \
                      PLOT_DIR + figureName, imageName, yLabel='RMSE', \
                      xLabel='Degree', SAVE_FIGURE=SAVE_PLOT)

figureName = 'MSEXOrder_test.jpg'
imageName = 'MSE in test set by polinomial order'
tr.PlotErrorFunction(xTest, mseTest, INITIAL_ORDER, FINAL_ORDER, \
                      PLOT_DIR + figureName, imageName, yLabel='RMSE', \
                      xLabel='Degree', SAVE_FIGURE=SAVE_PLOT)

figureName = 'RMSEXOrder_test.jpg'
imageName = 'RMSE in test set by polinomial order'
rmseTest = np.sqrt(mseTest)
tr.PlotErrorFunction(xTest, rmseTest, INITIAL_ORDER, FINAL_ORDER, \
                      PLOT_DIR + figureName, imageName, yLabel='RMSE', \
                      xLabel='Degree', SAVE_FIGURE=SAVE_PLOT)

pldPrices = mPLDSE.price
MIN_WINDOW_SIZE = 3
MAX_WINDOW_SIZE = pldPrices.size
mseMA = np.zeros((MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) + 1)

#Simple moving average trend extraction
for wSize in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
    mseMA[wSize - MIN_WINDOW_SIZE] = \
    tr.GetTrendMSEMovingAverage(pldPrices, wSize)


SAVE_PLOT = False
figureName = 'MSE_MA.jpg'
imageName = 'MSE for moving average using complete dataset'
xMA = np.arange(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1)
tr.PlotErrorFunction(xMA, mseMA, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1, \
                      PLOT_DIR + figureName, imageName, xLabel='Window Size', \
                      yLabel='RMSE', SAVE_FIGURE=SAVE_PLOT)

figureName = 'RMSE_MA.jpg'
imageName = 'RMSE for moving average using complete dataset'
rmseMA = np.sqrt(mseMA)
tr.PlotErrorFunction(xMA, rmseMA, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1, \
                      PLOT_DIR + figureName, imageName, xLabel='Window Size', \
                      yLabel='RMSE', SAVE_FIGURE=SAVE_PLOT)

#Exponential moving average trend extraction
MIN_WINDOW_SIZE = 3
MAX_WINDOW_SIZE = pldPrices.size
mseEMA = np.zeros((MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) + 1)

#Simple moving average trend extraction
for wSize in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
    mseEMA[wSize - MIN_WINDOW_SIZE] = \
    tr.GetTrendMSEExponentialMovingAverage(pldPrices, wSize)

SAVE_PLOT = False
figureName = 'MSE_EMA.jpg'
imageName = 'MSE for exponential moving average using complete dataset'
xMA = np.arange(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1)
tr.PlotErrorFunction(xMA, mseEMA, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1, \
                      PLOT_DIR + figureName, imageName, xLabel='Window Size', \
                      yLabel='RMSE', SAVE_FIGURE=SAVE_PLOT)


figureName = 'RMSE_EMA.jpg'
imageName = 'RMSE for exponential moving average using complete dataset'
rmseEMA = np.sqrt(mseEMA)
tr.PlotErrorFunction(xMA, rmseEMA, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1, \
                      PLOT_DIR + figureName, imageName, xLabel='Window Size', \
                      yLabel='RMSE', SAVE_FIGURE=SAVE_PLOT)


#Using SARIMA time series decomposion
MAX_ORDER = 2
INITIAL_TEST_DATE = '2018-01-01'
SAVE_PLOT = False
#param= (0, 2, 2)
#seasonal_param= (0, 2, 2)
tr.UseSARIMAToEstimateTemporalSeries(pldPrices, MAX_ORDER, \
                                     INITIAL_TEST_DATE, PLOT_DIR, \
#                                     param, \
#                                     seasonal_param, \
                                     SAVE_FIGURE = SAVE_PLOT)

#Time Series decomposition
figureName = 'TSA_additiveDecomposition.jpg'
decomposition = sm.tsa.seasonal_decompose(trainValidationSet, model='additive')
SAVE_PLOT = False
if SAVE_PLOT:
    fig = decomposition.plot()
    plt.show()
    plt.savefig(PLOT_DIR + figureName, bbox_inches='tight')
    
tsaTrend = decomposition.trend
tsaSeasonal = decomposition.seasonal
tsaResidual = decomposition.resid
tsaR = tsaResidual[~np.isnan(tsaResidual.price)]

#Residual Component Plot
SAVE_PLOT = False
SHOW_PLOT = False
title = 'FFT of Residual Signal'
X_LABEL = "Normalized Frequency"
Y_LABEL = "Magnitude"
figName = 'FFT_residual.jpg'

util.FFT(tsaR, X_LABEL, Y_LABEL, title, \
         figureName=PLOT_DIR + figName ,saveFig=SAVE_PLOT, showPlot=SHOW_PLOT)
    
#Residual Component Plot After removing senoidal cycles
fsm = FourierSeriesMinimizer()
fsm.SetRefenceValue(tsaR.price)
xTsaR = np.arange(0, tsaR.size, 1)
fsm.SetTimeVector(xTsaR)
initialGuess = [4.0, 0, 0.33]
minimizedCoef = fsm.MinimizeFourierSeriesCoefficients(initialGuess)
an, bn, fn = minimizedCoef[0]
nLinComponent = fsm.ExtractSenoidalInformation(an, bn, fn, originalSerie=tsaR.price)
title = "FFT of NonLinear Residue Signal"
figName = 'FFT_NonLinearresidual.jpg'
util.FFT(nLinComponent, X_LABEL, Y_LABEL, title, \
         figureName= PLOT_DIR + figName ,saveFig=SAVE_PLOT, showPlot=SHOW_PLOT)
