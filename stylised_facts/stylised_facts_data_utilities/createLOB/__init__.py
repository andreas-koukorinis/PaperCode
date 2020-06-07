import numpy as np
import pickle as pkl
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob

from scipy.signal import savgol_filter
from scipy.special import gamma

from collections import defaultdict

import self as self
from tqdm import tqdm

'''
Locations
'''

extHD = '/media/ak/My Passport/'
extHdData = "".join((extHD, 'Data'))
extHdExpData = "".join((extHD, 'Experiment Data'))  # ['features', 'labels', 'metrics', 'models']
extHdFutures = "".join((extHD, 'Barket Data/raw bloomberg data'))  # futures
futuresSymbols = []
cleanLOBFolder = "/".join((extHdExpData, 'CleanLOBData'))
barketData = '/media/ak/My Passport/Barket Data/'
#
targetDrive = barketData
bmrg_folders = [s for s in os.listdir(targetDrive) if s.endswith('Comdty')]
bmrg_trades = sorted([s for s in os.listdir(targetDrive) if s.endswith('y_trades')])
bmrg_quotes = sorted([s for s in os.listdir(targetDrive) if s.endswith('y_quotes')])
bmrg_tickers = sorted([bmrg_trades[idx].split('_t')[0] for idx, _ in enumerate(bmrg_trades)])


# '''
#
#   class to compute all the various quantities needed for the Limit Order Book
#   Tick, Volume, DV Bars calculationsyeah
#   butfairr u f''
#
#           args
#           df: pd.DataFrame()
#           price_column: name for price data
#           m: int(), threshold value for ticks
#           volume_column: name for volume data
#           vm : int(), threshold for volume
#       '''
class Clocks(object):

    def __init__(self, df, price_column, volume_column, dv_column, mticks, vticks, dvticks):
        self.df = df
        self.price_column = price_column
        self.mticks = mticks  # should this be a property of the class?
        self.volume_column = volume_column
        self.vticks = vticks
        self.dv_column = dv_column
        self.dvticks = dvticks
        pass

    def tick_bars(self):
        '''
        compute tick bars

        # args
            df: pd.DataFrame()
            column: name for price data
            m: int(), threshold value for ticks

        There are many ways to choose M, or the threshold value for sampling prices.
        One way is based on ratios of total dollar value/volume traded vs number of ticks.

        # returns
            idx: list of indices
        '''
        t = self.df[self.price_column]
        ts = 0
        idx = []
        for i, x in enumerate(tqdm(t)):
            ts += 1
            if ts >= self.mticks:
                idx.append(i)
                ts = 0
                continue
        return idx

    def tick_bar_df(self):

        idx = self.tick_bars()
        return self.df.iloc[idx].drop_duplicates()

    @staticmethod
    def mad_outlier(y, thresh=2.):
        '''
        compute outliers based on mad
        # args
            y: assumed to be array with shape (N,1)
            thresh: float()
        # returns
            array index of outliers
        '''
        median = np.median(y)
        diff = np.sum((y - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thresh

    def volume_bars(self):
        '''
        compute volume bars

        # args
            df: pd.DataFrame()
            volume_column: name for volume data
            m: int(), threshold value for volume
        # returns
            idx: list of indices
        '''
        t = self.df[self.volume_column]
        ts = 0
        idx = []
        for i, x in enumerate(tqdm(t)):
            ts += x
            if ts >= self.vticks:
                idx.append(i)
                ts = 0
                continue
        return idx

    def volume_bar_df(self):
        idx = self.volume_bars()
        return self.df.iloc[idx].drop_duplicates()

    def original_df(self):
        return self.df

    def dollar_bars(self):
        '''
        compute dollar bars

        # args
            df: pd.DataFrame()
            dv_column: name for dollar volume data
            m: int(), threshold value for dollars
        # returns
            idx: list of indices
        '''
        t = self.df[self.dv_column]
        ts = 0
        idx = []
        for i, x in enumerate(tqdm(t)):
            ts += x
            if ts >= self.dvticks:
                idx.append(i)
                ts = 0
                continue
        return idx

    def dollar_bar_df(self):
        idx = self.dollar_bars()
        return self.df.iloc[idx].drop_duplicates()


class autoCorrel(object):

    def __init__(self, df, tick_window=20):
        self.df = df
        self.tick_window = tick_window

    def ticks(self):
        ticks = np.sign(self.df.TradePrice - self.df.TradePrice.shift(1))
        ticks_adj = ticks.copy()
        return ticks_adj.replace(to_replace=0, method='ffill')

    def ema(self):
        '''
        returns exponential moving average of ticks, for a predefined window
        '''
        ema = self.ticks.ewm(span=self.tick_window).mean()
        return ema

    def figAutocorrel(self, title='autocorrelsymbol', xlineLevel=80):
        from matplotlib.ticker import AutoMinorLocator
        size = 15
        params = {'legend.fontsize': 'large',
                  'figure.figsize': (20, 8),
                  'axes.labelsize': size,
                  'axes.titlesize': size,
                  'xtick.labelsize': size * 0.75,
                  'ytick.labelsize': size * 0.75,
                  'axes.titlepad': 25}
        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        autoCorrelDict = {}
        # Set up the second axis
        ax2 = ax1.twiny()
        ax1.set_facecolor('xkcd:white')
        for i in np.arange(1, 500):
            ax1.bar(i, self.ticks().autocorr(lag=i), color='blue')
            autoCorrelDict[i] = self.ticks().autocorr(lag=i)
        ax1.set_title('Autocorrelation by lag')
        ax1.set_ylabel('Autocorrelation')
        ax1.set_xlabel('Lag')

        # Initial value of y at t=0, lifetime in s
        N, tau = 1, 15
        # Maximum time to consider (s)
        tmax = 500
        ntau = tmax // tau + 1
        # A suitable grid of time points, and the exponential decay itself
        t = np.linspace(0, tmax, 400)
        y = N * np.exp(-t / tau)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        ax1.plot(t, y, color='black', linestyle='--')
        #     axvline(x=.5, ymin=0.25, ymax=0.75)
        ax1.axvline(x=xlineLevel, ymin=0.0, ymax=0.8, ls='--', alpha=0.7, color='#334f8d')
        ax1.text(xlineLevel, 0.85, 'lag ' + str(xlineLevel), rotation=90, fontsize=12)
        xtick_labels = [r'$0$', r'$\tau$'] + [r'${}\tau$'.format(k) for k in range(2, ntau)]
        ax2.set_xticklabels(xtick_labels)
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + title + '.png', dpi=150)
        plt.show()
        return autoCorrelDict


def get_ohlc(ref, sub):
    '''
    fn: get ohlc from custom bars

    # args
        ref : reference pandas series with all prices
        sub : custom tick pandas series
    # returns
        tick_df : dataframe with ohlc values
    '''
    ohlc = []
    for i in tqdm(range(sub.index.shape[0] - 1)):
        start, end = sub.index[i], sub.index[i + 1]
        tmp_ref = ref.loc[start:end]
        max_px, min_px = tmp_ref.max(), tmp_ref.min()
        o, h, l, c = sub.iloc[i], max_px, min_px, sub.iloc[i + 1]
        ohlc.append((end, start, o, h, l, c))
    cols = ['End', 'Start', 'Open', 'High', 'Low', 'Close']
    return (pd.DataFrame(ohlc, columns=cols))


def rawLOBFIle(futuresFolder, symbolsFolder, symbolID, fileID):
    ''' read the appropriate data and produce a raw LOB file
    futuresFolder: where the extHDFutures folder is
    symbolID: which symbol to pick up
    symbolsFolder: folder with symbols. you use symbolID as an index in that folder
    '''
    listDatesFiles = os.listdir("/".join((futuresFolder, symbolsFolder[symbolID])))
    rawLOBFile = pd.read_csv("/".join((futuresFolder, symbolsFolder[symbolID], listDatesFiles[fileID])))

    return rawLOBFile


def createLOB(rawLOBFile):
    '''
    create a clean LOB based on the Bloomberg Files that I got from Barket.
    :param rawLOBFile: a raw LOB file that has bids, asks, sizes and trade quantities and levels-
    :return: LOB with dissagrated in various columns
    '''
    dfBID = rawLOBFile[rawLOBFile['type'] == 'BID']

    dfASK = rawLOBFile[rawLOBFile['type'] == 'ASK']
    dfTRADE = rawLOBFile[rawLOBFile['type'] == 'TRADE']

    dfBID.loc[:, ('TradeTime')] = pd.to_datetime(dfBID.time).values
    dfASK.loc[:, ('TradeTime')] = pd.to_datetime(dfASK.time).values
    dfTRADE.loc[:, ('TradeTime')] = pd.to_datetime(dfTRADE.time).values
    dfTRADE.loc[:, ('TradeId')] = dfTRADE.index.values
    dfBID = dfBID.dropna().fillna("ffill").sort_values('TradeTime')
    dfASK = dfASK.dropna().fillna("ffill").sort_values('TradeTime')
    dfTRADE = dfTRADE.dropna().fillna("ffill").sort_values('TradeTime')

    dfLOB = pd.merge_asof(dfBID, dfASK, on='TradeTime', allow_exact_matches=True).sort_values('TradeTime')

    LOB = pd.merge_asof(dfLOB, dfTRADE, on='TradeTime', allow_exact_matches=True)

    return pd.DataFrame(LOB)


def formatLOB(LOB):
    '''

   :param LOB: clean LOB dataframe
   :return: formatted with additional fields and more calculations
   '''
    LOB = LOB.rename(
        columns={'Unnamed: 0_y': 'AskQuoteId', 'Unnamed: 0_x': 'BidQuoteId', 'value_x': 'BestBid', 'value_y': 'BestAsk',
                 'size_x': 'BidSize', 'size_y': 'AskSize', 'time_x': 'QuoteTime', 'time': 'TradedTime',
                 'value': 'TradePrice', 'size': 'TradeVolume', 'Unnamed: 0': 'TradeId'})
    LOB.BidSize = LOB['BidSize'].replace(0, 1)
    LOB.AskSize = LOB['AskSize'].replace(0, 1)
    LOB['TimeStamp'] = pd.to_datetime(LOB.TradeTime).dt.time
    LOB['TradeVolume'] = LOB['TradeVolume'].fillna(0)
    LOB['milliSeconds'] = [(((x.hour * 60 + x.minute) * 60 + x.second) * 1000) for x in LOB['TimeStamp']]
    LOB['DollarVolume'] = LOB.TradePrice * LOB.TradeVolume
    LOB['MicroPrice'] = (LOB.BestAsk * LOB.AskSize + LOB.BestBid * LOB.BidSize) / (
                LOB.AskSize + LOB.BidSize)  # weighted mid price
    LOB['TradeSize'] = LOB.TradeVolume

    return LOB


def calcLOB(LOB):
    '''

    :param LOB: takes in a dataframe and computes various concepts like Duration, Price Change, Log Return, Dollar Volume Micro Price
    we are arddubg sine if of the same calculations above so I am able to work through the applications on the different activity clocks
    :return: LB data frame with the new calculations
    '''

    LOB['Duration'] = LOB.TradeTime.diff().apply(lambda tt: tt.total_seconds()) * 1000
    LOB['Duration'] = LOB['Duration'].fillna(0)
    LOB['PriceChange'] = LOB.TradePrice.pct_change()
    LOB['LogReturn'] = np.log(1 + LOB.PriceChange)
    LOB['BidOffer'] = LOB.BestAsk - LOB.BestBid
    LOB['DollarVolume'] = LOB.TradePrice * LOB.TradeVolume
    LOB['MicroPrice'] = (LOB.BestAsk * LOB.AskSize + LOB.BestBid * LOB.BidSize) / (
                LOB.AskSize + LOB.BidSize)  # weighted mid price
    LOB['MicroPricePctChange'] = LOB['MicroPrice'].pct_change()
    LOB['FwdMPChange_1'] = LOB.MicroPricePctChange.shift(1)
    LOB['FwdMPChange_5'] = LOB.MicroPricePctChange.shift(5)
    LOB['LogTradeSize'] = np.log(LOB.TradeVolume)

    return LOB


def storeCleanLOB(cleanLOB, targetFolder, symbolID, symbolsFolder=bmrg_folders):
    """
    meant to store the file into a hard drive
    :param cleanLOB: a clean LOB to be stored
    :param targetFolder: where do you want it stored, defaults in the cleanLOBFolder in the ext HD
    :param symbolsFolder: defaults to bmrg Folders
    :param symbolID: enumerate of symbol
    :return: stores the file
    """
    symbol = symbolsFolder[symbolID].split("_")[0]  # pick the right symbol
    targetFolder = cleanLOBFolder
    cleanLOBDate = str(pd.to_datetime(cleanLOB.QuoteTime[0]).date())  # pick the right date
    cleanLOBDateFileName = "_".join(('LOB', str(symbol), cleanLOBDate + ".csv"))
    cleanLOBFileLoc = "/".join((cleanLOBFolder, symbol, cleanLOBDateFileName))
    return cleanLOB.to_csv(cleanLOBFileLoc)


class Volestim(object):
    def __init__(self, df, clean=True, window=30, trading_periods=252):
        '''
        realised volatility estimation using different methodologies
        :param df:
         Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        clean : boolean
            Set to True to remove the NaNs at the beginning of the series

        Returns
        -------
        y : pandas.DataFrame
            Estimator series values

        '''
        self.df = df
        self.window = window
        self.trading_periods = trading_periods
        self.clean = True
        pass

    def rogersSatchell(self):

        log_ho = (self.df['High'] / self.df['Open']).apply(np.log)
        log_lo = (self.df['Low'] / self.df['Open']).apply(np.log)
        log_co = (self.df['Close'] / self.df['Open']).apply(np.log)

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        def f(v):
            return self.trading_periods * v.mean() ** 0.5

        result = rs.rolling(
            window=self.window,
            center=False).apply(func=f)

        if self.clean:
            return result.dropna()
        else:
            return result

    def kurtosis(self):

        log_return = (self.df['Close'] / self.df['Close'].shift(1)).apply(np.log)

        result = log_return.rolling(
            window=self.window,
            center=False
        ).kurt()

        if self.clean:
            return result.dropna()
        else:
            return result

    def yangZhang(self):

        log_ho = (self.df['High'] / self.df['Open']).apply(np.log)
        log_lo = (self.df['Low'] / self.df['Open']).apply(np.log)
        log_co = (self.df['Close'] / self.df['Open']).apply(np.log)

        log_oc = (self.df['Open'] / self.df['Close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc ** 2

        log_cc = (self.df['Close'] / self.df['Close'].shift(1)).apply(np.log)
        log_cc_sq = log_cc ** 2

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        close_vol = log_cc_sq.rolling(
            window=self.window,
            center=False
        ).sum() * (1.0 / (self.window - 1.0))
        open_vol = log_oc_sq.rolling(
            window=self.window,
            center=False
        ).sum() * (1.0 / (self.window - 1.0))
        window_rs = rs.rolling(
            window=self.window,
            center=False
        ).sum() * (1.0 / (self.window - 1.0))

        k = 0.34 / (1 + (self.window + 1) / (self.window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(self.trading_periods)

        if self.clean:
            return result.dropna()
        else:
            return result

    def garmanKlass(self):

        log_hl = (self.df['High'] / self.df['Low']).apply(np.log)
        log_co = (self.df['Close'] / self.df['Open']).apply(np.log)

        rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

        def f(v):
            return (self.trading_periods * v.mean()) ** 0.5

        result = rs.rolling(window=self.window, center=False).apply(func=f)

        if self.clean:
            return result.dropna()
        else:
            return result

    def simpleVol(self):

        log_return = (self.df['Close'] / self.df['Close'].shift(1)).apply(np.log)

        result = log_return.rolling(
            window=self.window,
            center=False
        ).std() * np.sqrt(self.trading_periods)

        if self.clean:
            return result.dropna()
        else:
            return result

    def parkinson(self):

        rs = (1.0 / (4.0 * np.log(2.0))) * ((self.df['High'] / self.df['Low']).apply(np.log)) ** 2.0

        def f(v):
            return self.trading_periods * v.mean() ** 0.5

        result = rs.rolling(
            window=self.window,
            center=False
        ).apply(func=f)

        if self.clean:
            return result.dropna()
        else:
            return result

    def hodgesTompkins(self):

        log_return = (self.df['Close'] / self.df['Close'].shift(1)).apply(np.log)

        vol = log_return.rolling(
            window=self.window,
            center=False
        ).std() * np.sqrt(self.trading_periods)

        h = self.window
        n = (log_return.count() - h) + 1

        adj_factor = 1.0 / (1.0 - (h / n) + ((h ** 2 - 1) / (3 * n ** 2)))

        result = vol * adj_factor

        if self.clean:
            return result.dropna()
        else:
            return result


    @staticmethod
    def tripower_volatility(dframe, columnName,rollingWindow=50):
        """
        Realized tripower volatility (e.g. Barndorff-Nielsen, Shephard, and Winkel (2006))
        """
        x = pd.Series(dframe[str(columnName)])
        xi = 0.5 * (gamma(5 / 6) / gamma(1 / 2)) ** -3
        z = (x.abs() ** (2 / 3) * x.shift(1).abs() ** (2 / 3) * x.shift(-1).abs() ** (2 / 3)).bfill().ffill()
        return xi * z.rolling(window=rollingWindow).sum()

    @staticmethod
    def shortest_half(dframe, columnName,rollingWindow=50):
        """
        Shortest-half scale estimator (Rousseeuw and Leroy, 1998)
        """
        xs = np.sort(dframe[str(columnName)])
        l = dframe[str(columnName)].size
        h = int(np.floor(l / 2) + 1)
        if l % 2 == 0:
            sh = 0.7413 * np.min(xs[h - 1:] - xs[:h - 1])
        else:
            sh = 0.7413 * np.min(xs[h - 1:] - xs[:h])
        return sh


    @staticmethod
    def bipower_variation(dframe, columnName,rollingWindow=50):
        '''
        Bipower Variation (BV) is the sum of the product of absolute time series returns
        :param column: price column
        :return: returns bivariate variation
        '''
        X = dframe[str(columnName)]
        u = np.sqrt(np.pi / 2) ** -2
        pre_log = u * sum([abs(f) * abs(p) for f, p in zip(X[2:], X[1:])])

        # check for log(0)
        try:
            BV = np.log(pre_log)
        except:
            BV = np.nan

        return BV
