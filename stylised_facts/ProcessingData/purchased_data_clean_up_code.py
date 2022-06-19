import pandas as pd
import pickle
import os
import glob

inPath = '/media/ak/DataOnly/OB'
symbol = ['CL']
outPath = '/media/ak/DataOnly/FuturesData'

if __name__ == '__main__':
    symbolIdx = 0
    clf_path = os.path.join(inPath, 'CL')[symbolIdx]
    for idx in range(len(os.listdir(clf_path))):
        fileToLoad = pd.read_csv(os.path.join(inPath, symbol[0], os.listdir(clf_path)[idx]))
        # subsets = [os.path.join(inPath, symbol[0], CLFiles[idx]))[['Date', 'Time', 'L1-BidPrice', 'L1-BidSize', 'L1-BuyNo',
    #    'L1-AskPrice', 'L1-AskSize', 'L1-SellNo']].dropna() for idx in range(len(CLFiles))]
