import numpy as np
import pickle as pkl
import os

import pandas as pd
import numpy as np

import time

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
cleanLOBFolder = "/".join((extHdExpData,'CleanLOBData'))
barketData = '/media/ak/My Passport/Barket Data/'
#
targetDrive = barketData
bmrg_folders=[s for s in os.listdir(targetDrive ) if s.endswith('Comdty')]
bmrg_trades=sorted([s for s in os.listdir(targetDrive ) if s.endswith('y_trades')])
bmrg_quotes=sorted([s for s in os.listdir(targetDrive ) if s.endswith('y_quotes')])
bmrg_tickers=sorted([bmrg_trades[idx].split('_t')[0] for idx,_ in enumerate(bmrg_trades)])
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
        self.mticks = mticks # should this be a property of the class?
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

    dfBID.loc[:,('TradeTime')] = pd.to_datetime(dfBID.time).values
    dfASK.loc[:,('TradeTime')] = pd.to_datetime(dfASK.time).values
    dfTRADE.loc[:,('TradeTime')] = pd.to_datetime(dfTRADE.time).values
    dfTRADE.loc[:,('TradeId')] = dfTRADE.index.values
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
   LOB['TradeVolume'] = LOB['TradeVolume'] .fillna(0)
   LOB['milliSeconds'] = [(((x.hour * 60 + x.minute) * 60 + x.second) * 1000) for x in LOB['TimeStamp']]
   LOB['DollarVolume'] = LOB.TradePrice * LOB.TradeVolume
   LOB['MicroPrice'] = (LOB.BestAsk * LOB.AskSize + LOB.BestBid * LOB.BidSize) / (LOB.AskSize + LOB.BidSize)  # weighted mid price
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
    LOB['MicroPrice'] = (LOB.BestAsk * LOB.AskSize + LOB.BestBid * LOB.BidSize) / (LOB.AskSize + LOB.BidSize) #weighted mid price
    LOB['MicroPricePctChange'] = LOB['MicroPrice'].pct_change()
    LOB['FwdMPChange_1'] = LOB.MicroPricePctChange.shift(1)
    LOB['FwdMPChange_5'] = LOB.MicroPricePctChange.shift(5)
    LOB['LogTradeSize'] = np.log(LOB.TradeVolume)

    return LOB


def storeCleanLOB( cleanLOB, targetFolder,symbolID, symbolsFolder= bmrg_folders ):
    """
    meant to store the file into a hard drive
    :param cleanLOB: a clean LOB to be stored
    :param targetFolder: where do you want it stored, defaults in the cleanLOBFolder in the ext HD
    :param symbolsFolder: defaults to bmrg Folders
    :param symbolID: enumerate of symbol
    :return: stores the file
    """
    symbol = symbolsFolder[symbolID].split("_")[0] #pick the right symbol
    targetFolder = cleanLOBFolder
    cleanLOBDate = str(pd.to_datetime(cleanLOB.QuoteTime[0]).date()) #pick the right date
    cleanLOBDateFileName = "_".join(('LOB', str(symbol), cleanLOBDate + ".csv"))
    cleanLOBFileLoc = "/".join((cleanLOBFolder,symbol ,cleanLOBDateFileName))
    return cleanLOB.to_csv(cleanLOBFileLoc)

