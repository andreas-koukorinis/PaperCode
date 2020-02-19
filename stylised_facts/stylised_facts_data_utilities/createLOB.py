import numpy as np
import pickle
import os

import pandas as pd
import numpy as np

import time

from collections import defaultdict

'''
Locations
'''

extHD = '/media/ak/My Passport/'
extHdData = "".join((extHD, 'Data'))
extHdExpData = "".join((extHD, 'Experiment Data'))  # ['features', 'labels', 'metrics', 'models']
extHdFutures = "".join((extHD, 'Barket Data/raw bloomberg data'))  # futures
futuresSymbols = []
cleanLOBFolder = "/".join((extHdExpData,'CleanLOBData'))
bmrg_folders = [s for s in os.listdir(extHdFutures) if ('Comdty') in s]


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
    :param rawLOBFile:
    :return: LOB
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
                'value': 'TradePrice', 'size': 'TradeSize', 'Unnamed: 0': 'TradeId'})
   LOB.BidSize = LOB['BidSize'].replace(0, 1)
   LOB.AskSize = LOB['AskSize'].replace(0, 1)
   LOB['TimeStamp'] = pd.to_datetime(LOB.TradeTime).dt.time
   LOB['milliSeconds'] = [(((x.hour * 60 + x.minute) * 60 + x.second) * 1000) for x in LOB['TimeStamp']]
   LOB['Duration'] = LOB.TradeTime.diff().apply(lambda tt: tt.total_seconds()) * 1000
   LOB['PriceChange'] = LOB.TradePrice.pct_change()
   LOB['LogReturn'] = np.log(1 + LOB.PriceChange)
   LOB['BidOffer'] = LOB.BestAsk - LOB.BestBid
   LOB['MicroPrice'] = (LOB.BestAsk * LOB.AskSize + LOB.BestBid * LOB.BidSize) / (LOB.AskSize + LOB.BidSize)
   LOB['MicroPricePctChange'] = LOB['MicroPrice'].pct_change()
   LOB['FwdMPChange_1'] = LOB.MicroPricePctChange.shift(1)
   LOB['FwdMPChange_5'] = LOB.MicroPricePctChange.shift(5)
   LOB['LogTradeSize'] = np.log(LOB.TradeSize)

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



if __name__ == '__main__':
    symbolID=7
    for symbolID in range(0, len(bmrg_folders)):
        print(bmrg_folders[symbolID])



    for fileIDx in range(0,200):
        fileID=fileIDx
        rawInputFile =rawLOBFIle(futuresFolder=extHdFutures, symbolsFolder=bmrg_folders, symbolID=symbolID, fileID=fileID)
        testLOB =(createLOB(rawLOBFile = rawInputFile))
        cleanLOB = formatLOB(testLOB)
        storeCleanLOB(cleanLOB,cleanLOBFolder, symbolID, bmrg_folders)