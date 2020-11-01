import pandas as pd
import numpy as np
import os
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
import pickle as pkl
import fnmatch
import datetime as dt
from datetime import timedelta
import glob

# Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt


def listdirs(folder):  # return only directories from a master folder
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def find_path(basepath, stringname):
    for fname in os.listdir(basepath):
        path = os.path.join(basepath, fname)
        if os.path.isdir(path):
            if stringname in fname:
                return path


def get_ticket(rx_path):
    ticker = rx_path.split(".")[0].split("/")[-1].split("-")[0]
    return ticker


def ticker_trades_dir(ticker):
    dest = os.path.join(os.getenv('FINANCE_DATA'), "_".join((ticker, 'trades')))
    if not os.path.isdir(dest):
        os.makedirs(dest)


def quotes_trades_dir(ticker):
    dest = os.path.join(os.getenv('FINANCE_DATA'), "_".join((ticker, 'quotes')))
    if not os.path.isdir(dest):
        os.makedirs(dest)


def agg_on_trd_time(gr):
    """
    Utility func to aggregate trades on timestamp. All trades with equal time stamp
    will collapse to one row and the traded price will be the volume weighted traded
    price.
    """
    vTrdPrice = np.sum(gr['TradedPrice'] * gr['Volume']) / np.sum(gr['Volume'])
    volume = np.sum(gr['Volume'])

    return pd.Series({'Volume': volume,
                      'TradedPrice': vTrdPrice})


def open_pickle_file(path, pickle_file):
    file_loc = os.path.join(path, pickle_file)
    pickle_to_file = pkl.load(open(file_loc, "rb"))
    return pickle_to_file


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)

    # check length
    if len(a_set.intersection(b_set)) > 0:
        return a_set.intersection(b_set)
    else:
        return "no common elements"


def obv_calc(df):
    df['SignedVolume'] = df['Volume'] * np.sign(df['TradedPrice'].diff()).cumsum()
    df['SignedVolume'][:1] = 0
    df['OBV'] = df['SignedVolume'].cumsum()
    df = df.drop(columns=['SignedVolume'])
    return df


def chaikin_mf(df, period=5):
    df["MF Multiplier"] = (df['TradedPrice'] - (df['TradedPrice'].expanding(period).min()) \
                           - (df['TradedPrice'].expanding(period).max() - df['TradedPrice'])) / (
                                  df['TradedPrice'].expanding(period).max() - df['TradedPrice'].expanding(
                              period).min())
    df["MF Volume"] = df['MF Multiplier'] * df['Volume']
    df['CMF'] = df['MF Volume'].sum() / df["Volume"].rolling(5).sum()
    df = df.drop(columns=['MF Multiplier', 'MF Volume'])
    return df

##locations ##
dataOnlyDrive = ('/media/ak/DataOnly')
cleanBloombergDataLocation = os.path.join(dataOnlyDrive, 'CleanBloombergData')
targetDrive = dataOnlyDrive
bmrg_folders=[s for s in os.listdir(targetDrive ) if s.endswith('Comdty')]
bmrg_trades=sorted([s for s in os.listdir(targetDrive ) if s.endswith('y_trades')])
bmrg_quotes=sorted([s for s in os.listdir(targetDrive ) if s.endswith('y_quotes')])
bmrg_tickers=sorted([bmrg_trades[idx].split('_t')[0] for idx,_ in enumerate(bmrg_trades)])

if __name__ == '__main__':
    for symbolIdx in range(1,2):
        print(bmrg_quotes[symbolIdx])
        print(bmrg_trades[symbolIdx])
        # get dates and files
        symbol_quotes = os.path.join(targetDrive, str(bmrg_quotes[symbolIdx]))
        symbol_trades = os.path.join(targetDrive, str(bmrg_trades[symbolIdx]))
        symbolQuoteDates = [quoteFile.split(".")[0] for quoteFile in os.listdir(symbol_quotes)]
        symbolTradeDates = [tradeFile.split(".")[0] for tradeFile in os.listdir(symbol_trades)]
        # create dictionaries that have all the data we may need/want. on for top of the LOB and one for trades
        quoteTradeDates = [eventDate for eventDate in symbolQuoteDates if eventDate in symbolTradeDates]
        len(quoteTradeDates)
        dfAllTrades = {}
        dfAllQuotes = {}
        for idx, date in enumerate(quoteTradeDates):
            tradeTestDate = pd.read_csv(os.path.join(symbol_trades, quoteTradeDates[idx] + '.csv'))
            tradeQuoteDate = pd.read_csv(os.path.join(symbol_quotes, quoteTradeDates[idx] + '.csv'))
            dfAllTrades[date] = tradeTestDate
            dfAllQuotes[date] = tradeQuoteDate
        dfAllQuotesName = "".join(('AllQuotes', bmrg_quotes[symbolIdx], 'Comdty.pkl'))
        dfAllTradesName = "".join(('AllTrades', bmrg_trades[symbolIdx], 'Comdty.pkl'))
        pkl.dump(dfAllQuotes, open("/".join((cleanBloombergDataLocation, dfAllQuotesName)), "wb"))
        pkl.dump(dfAllTrades, open("/".join((cleanBloombergDataLocation, dfAllTradesName)), "wb"))

        dQuoteKeys = list(dfAllQuotes.keys())  # one common set of keys at the moment
        dTradeKeys = list(dfAllTrades.keys()) # list of all trades
        commonDates = list(set(dQuoteKeys).intersection(set(dTradeKeys)))

        dfLOBRaw = {}
        # QuoteColumns =['Unnamed: 0','TimeStamp']
        # TradeColumns= ['Unnamed: 0', 'size', 'time', 'type', 'value']
        for idx, dateKey in enumerate(commonDates):
            # dfAllQuotes[dKeys[idx]].reset_index(level=0, inplace=True)
            dfAllQuotes[commonDates[idx]]['TimeStamp'] = pd.to_datetime(dfAllQuotes[commonDates[idx]]['time'])
            dfAllTrades[commonDates[idx]]['TradeTimeStamp'] = pd.to_datetime(dfAllTrades[commonDates[idx]]['time'])
            dfAllTrades[commonDates[idx]]['TradedPrice'] = dfAllTrades[commonDates[idx]]['value']
            dfAllTrades[commonDates[idx]]['TradedSize'] = dfAllTrades[commonDates[idx]]['size']
            dfAllTrades[commonDates[idx]].rename(
                columns={'type': 'QuotedSide', 'value': 'bestPrice', 'size': 'QuoteSize', 'time': 'QuoteTimeStamp'},
                inplace=True)
            dfAllTrades[commonDates[idx]]['Duration'] = dfAllTrades[
                                                            commonDates[idx]].TradeTimeStamp.diff() / np.timedelta64(1,
                                                                                                                     'ms')
            # dfBID = dfAllQuotes[commonDates[idx]][dfAllQuotes[commonDates[idx]]['type'] == 'BID'].rename(
            #     columns={'type': 'BidSide', 'value': 'bestBidPrice', 'size': 'bestBidSize',
            #              'time': 'TimeStampS'})  # .drop(QuoteColumns, inplace=True, axis=1)
            # dfASK = dfAllQuotes[commonDates[idx]][dfAllQuotes[commonDates[idx]]['type'] == 'ASK'].rename(
            #     columns={'type': 'AskSide', 'value': 'bestAskPrice', 'size': 'bestAskSize',
            #              'time': 'TimeStampS'})  # .drop(QuoteColumns, inplace=True, axis=1)
            # dfLOB = dfBID.merge(dfASK, left_on='TimeStampS', right_on='TimeStampS')
            # dfLOB = dfLOB.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'AskSide', 'BidSide'], axis=1)
            dfAllTrades[commonDates[idx]].TradeTimeStamp.diff().dropna() / np.timedelta64(1,
                                                                                          'ms')  # dfLOB =dfBID.merge(dfASK,left_on='TimeStampS', right_on='TimeStampS')
            # dfLOB= dfLOB.drop(['Unnamed: 0_x', 'Unnamed: 0_y','AskSide','BidSide'], axis=1)
            dfAllTrades[commonDates[idx]]['Duration'] = dfAllTrades[commonDates[
                idx]].TradeTimeStamp.diff().dropna() / np.timedelta64(1, 'ms')
            dfAllQuotesName = "".join(('AllQuotes', bmrg_quotes[symbolIdx], 'symbol.pkl'))
            dfAllTradesName = "".join(('AllTrades', bmrg_trades[symbolIdx], 'symbol.pkl'))
            # dfLOBName = "".join(('LOB', bmrg_trades[symbolIdx], 'symbol.pkl'))
        pkl.dump(dfAllQuotes, open("/".join((cleanBloombergDataLocation, dfAllQuotesName)), "wb"))
        pkl.dump(dfAllTrades, open("/".join((cleanBloombergDataLocation, dfAllTradesName)), "wb"))
            # pkl.dump(dfLOB, open("/".join((cleanBloombergDataLocation, dfLOB)), "wb"))


