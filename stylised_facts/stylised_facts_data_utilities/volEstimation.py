import os
import createLOB
from collections import defaultdict
import time
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from powerlaw import plot_pdf, Fit, pdf
OHLCData= '/media/ak/My Passport/Experiment Data/OHLCData/'
estimFolder = '/media/ak/My Passport/Experiment Data/RealisedVolatilityEstimation/'

OHLCDataList = os.listdir(OHLCData)

symbols =['FB1','JB1','FV1','G_1']


def plotTimeSeries(estiChoice, ohlcDF, window=100, rollingWindow=1000, symbolID='symbol'):
    if estiChoice == 1:
        estimator = createLOB.Volestim(ohlcDF, window=window).hodgesTompkins()
    elif estiChoice == 2:
        estimator = createLOB.Volestim(ohlcDF, window=window).yangZhang()
    elif estiChoice == 3:
        estimator = createLOB.Volestim(ohlcDF, window=window).parkinson()
    elif estiChoice == 4:
        estimator = createLOB.Volestim(ohlcDF, window=window).garmanKlass()
    else:
        estimator = createLOB.Volestim(ohlcDF, window=window).rogersSatchell()

    date = estimator.index

    top_q = estimator.rolling(window=rollingWindow, center=False).quantile(quantiles[1])
    median = estimator.rolling(window=rollingWindow, center=False).median()
    bottom_q = estimator.rolling(window=rollingWindow, center=False).quantile(quantiles[0])
    realized = estimator.ewm(com=rollingWindow).mean()
    last = estimator.iloc[-1]
    f = lambda x: "%i%%" % round(x * 10000, 0)
    # figure
    size = 15
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    fig = plt.figure(figsize=(16, 9))
    fig.autofmt_xdate()
    left, width = 0.07, 0.65
    bottom, height = 0.2, 0.7
    left_h = left + width + 0.02

    rect_cones = [left, bottom, width, height]
    rect_box = [left_h, bottom, 0.17, height]

    cones = plt.axes(rect_cones)
    cones.set_facecolor("white")
    box = plt.axes(rect_box)
    # set the plots
    cones.plot(date, top_q, linestyle='-', label=str(int(quantiles[1] * 100)) + " %ile")
    cones.plot(date, median, linestyle='--', label="Median")
    cones.plot(date, bottom_q, linestyle='-', label=str(int(quantiles[0] * 100)) + " %ile")
    cones.plot(date, realized, color='lightblue', linestyle='--', label="Realized")

    # set and format the y-axis labels
    locs = cones.get_yticks()
    cones.set_yticklabels(map(f, locs))

    # turn on the grid
    cones.grid(True, axis='y', which='major', alpha=0.5)
    # set the legend
    cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # box plots
    box.boxplot(realized, notch=1, sym='+')
    box.plot(1, last, 'b+')

    # set and format the y-axis labels
    locs = box.get_yticks()
    box.set_yticklabels(map(f, locs))

    # move the y-axis ticks on the right side
    box.yaxis.tick_right()

    # turn on the grid
    box.grid(True, axis='y', which='major', alpha=0.5)

    if estiChoice == 1:
        cones.set_title('Hodges-Tompkings realised volatility series. Symbol: ' + str(symbolID))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'HodgesTompkinsTimeSeries' + str(symbolID)
                    + 'window' + str(window) + 'rollingWindow' + str(rollingWindow) + '.png', dpi=150)
    if estiChoice == 2:
        cones.set_title('Yang-Zhang realised volatility series. Symbol: ' + str(symbolID))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'YangZhangTimeSeries' + str(symbolID)
                    + 'window' + str(window) + 'rollingWindow' + str(rollingWindow) + '.png', dpi=150)
    if estiChoice == 3:
        cones.set_title('Parkinson realised volatility series. Symbol: ' + str(symbolID))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'ParkinsonTimeSeries' + str(symbolID)
                    + 'window' + str(window) + 'rollingWindow' + str(rollingWindow) + '.png', dpi=150)
    if estiChoice == 4:
        cones.set_title('Garman-Klass realised volatility series Symbol: ' + str(symbolID))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'GarmanKlassTimeSeries' + str(symbolID)
                    + 'window' + str(window) + 'rollingWindow' + str(rollingWindow) + '.png', dpi=150)
    if estiChoice == 5:
        cones.set_title('Rogers-Saatchell realised volatility series Symbol: ' + str(symbolID))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'RogSatchellTimeSeries' + str(symbolID)
                    + 'window' + str(window) + 'rollingWindow' + str(rollingWindow) + '.png', dpi=150)

    plt.show()

    return fig, plt


def PlotCones(estiChoice, ohlcDF, symbol):
    '''
    1: hodges Tompkins
    2: yang Zhang
    3: Parkinson
    4: Garman Klass
    '''

    max_ = []
    min_ = []
    top_q = []
    median = []
    bottom_q = []
    realized = []
    data = []

    windows = [10, 30, 60, 90]
    quantiles = [0.25, 0.75]
    for window in windows:
        if estiChoice == 1:
            estimator = createLOB.Volestim(ohlcDF, window=window).hodgesTompkins()
        elif estiChoice == 2:
            estimator = createLOB.Volestim(ohlcDF, window=window).yangZhang()
        elif estiChoice == 3:
            estimator = createLOB.Volestim(ohlcDF, window=window).parkinson()
        elif estiChoice == 4:
            estimator = createLOB.Volestim(ohlcDF, window=window).garmanKlass()
        else:
            estimator = createLOB.Volestim(ohlcDF, window=window).rogersSatchell()

        max_.append(estimator.max())
        top_q.append(estimator.quantile(quantiles[1]))
        median.append(estimator.median())
        bottom_q.append(estimator.quantile(quantiles[0]))
        min_.append(estimator.min())
        data.append(estimator)
        realized.append(estimator[-1:])

    #     size = 15
    #     params = {'legend.fontsize': 'large',
    #               'figure.figsize': (20, 8),
    #               'axes.labelsize': size,
    #               'axes.titlesize': size,
    #               'xtick.labelsize': size * 0.75,
    #               'ytick.labelsize': size * 0.75,
    #               'axes.titlepad': 25
    fig = plt.figure(figsize=(16, 9))
    # plt.xlabel('Horizon of Volatility Estimation')

    fig.set_facecolor('white')
    fig.autofmt_xdate()
    left, width = 0.07, 0.65
    bottom, height = 0.2, 0.7
    left_h = left + width + 0.02
    rect_cones = [left, bottom, width, height]
    rect_box = [left_h, bottom, 0.17, height]
    cones = plt.axes(rect_cones)
    box = plt.axes(rect_box)
    realized = list(np.asarray(pd.DataFrame(realized)).flatten())
    # set the plots
    cones.plot(windows, max_, label="Max")
    cones.plot(windows, top_q, label=str(int(quantiles[1] * 100)) + " percentile")
    cones.plot(windows, median, label="Median", linewidth=3.0)
    cones.plot(windows, bottom_q, label=str(int(quantiles[0] * 100)) + " percentile")
    cones.plot(windows, min_, label="Min")
    cones.plot(windows, realized, 'r-.', label="Realized", linewidth=2.0)

    # set the x ticks and limits
    cones.set_xticks(windows)
    cones.set_xlim((windows[0] - 5, windows[-1] + 5))

    # set and format the y-axis labels
    if estiChoice == 1:
        f = lambda x: "%i%%" % round(x * 100, 0)

    elif estiChoice == 2:
        f = lambda x: "%i%%" % round(x * 100, 0)
    elif estiChoice == 3:
        f = lambda x: "%i%%" % round(x * 100, 0)
    elif estiChoice == 4:
        f = lambda x: "%i%%" % round(x * 1000, 0)
    else:
        f = lambda x: "%i%%" % round(x * 100, 0)
    locs = cones.get_yticks()
    cones.set_yticklabels(map(f, locs))

    # turn on the grid
    cones.grid(True, axis='y', which='major', alpha=0.5)

    # set the title
    # cones.set_title(self._estimator + ' (' + self._symbol + ', daily ' + self._start + ' to ' + self._end + ')')
    cones.set_xlabel('Windows')
    cones.set_ylabel('Realised Volatility')

    # set the legend
    cones.legend(loc='upper left', bbox_to_anchor=(-0.003, 1.0), ncol=3)
    # box plot
    box.boxplot(data, notch=1, sym='+')
    box.plot([i for i in range(1, len(windows) + 1)], realized, color='r', marker='*', markeredgecolor='k')

    # set and format the y-axis labels
    locs = box.get_yticks()
    box.set_yticklabels(map(f, locs))

    # move the y-axis ticks on the right side
    box.yaxis.tick_right()

    # turn on the grid
    box.grid(True, axis='y', which='major', alpha=0.5)

    if estiChoice == 1:
        cones.set_title('Hodges-Tompkins realised volatility estimation (volatility cones) for symbol:  ' +str(symbol))

        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'HodgesTompkinsVolCones' + '.png', dpi=150)

    if estiChoice == 2:
        cones.set_title('Yang-Zhang realised volatility estimation (volatility cones) for symbol:  ' +str(symbol))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'YangZhangVolCones'+ str(symbol) + '.png', dpi=150)

    if estiChoice == 3:
        cones.set_title('Parkinson realised volatility estimation (volatility cones) for symbol:  ' +str(symbol))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'ParkinsonVolCones'+ str(symbol)  + '.png', dpi=150)
    if estiChoice == 4:
        cones.set_title('Garman-Klass realised volatility estimation (volatility cones) for symbol: ' +str(symbol))
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'GarmanKlassCones'+ str(symbol) +'.png', dpi=150)
    plt.show()

    return fig, plt


if __name__ == '__main__':
    for symbolIdx, symbol in enumerate(symbols):
        print(symbol)

        pklList = list(np.sort([s for s in OHLCDataList if (str(symbols[symbolIdx])) in s and ('ohlcFile') in s]))
        combinedDF = pd.DataFrame()
        for pklIdx, _ in enumerate(pklList):

            print(pklIdx)
            pklOHLCFile = "".join((OHLCData, pklList[pklIdx]))
            print(pklOHLCFile)
            combinedDF = combinedDF.append(pickle.load(open(pklOHLCFile, "rb")))

        combinedDFLoc = "/".join((OHLCData,str(symbol)+'CombinedDF.csv'))
        combinedDF.to_csv(combinedDFLoc, index=False, header=True)
        print(combinedDF.head(3))
        print('has been saved in :', combinedDFLoc)
        #
        # PlotCones(estiChoice=1, ohlcDF=combinedDF, symbol=symbol)
        #
        # PlotCones(estiChoice=2, ohlcDF=combinedDF, symbol=symbol)
        #
        # PlotCones(estiChoice=3, ohlcDF=combinedDF, symbol=symbol)
        #
        # PlotCones(estiChoice=4, ohlcDF=combinedDF, symbol=symbol)

        PlotCones(estiChoice=5, ohlcDF=combinedDF, symbol=symbol)


