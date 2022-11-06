 #import lob_for_futures as lob
import os
from collections import defaultdict
import pickle
import os
import numpy as np
import pickle as pkl
import pandas as pd
import pickle
from collections import defaultdict
import sys
from scipy.stats import jarque_bera

sys.path.append('/home/ak/Documents/Research/PaperCode/stylised_facts')
import lob_for_futures as lob

figures_destination = '/home/ak/Documents/Research/Papers/StylisedFactsPaper/figures'
elements = '/media/ak/Elements/OrderBookProcessedData'
symbols = sorted(os.listdir(elements))


####
def open_pickle_filepath(pickle_file):
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file


bars = ['volume_bar', 'calendar_bar', 'usd_volume_bar', 'tick_bar']

if __name__ == '__main__':
    symbolIdx = 1 #pick a symbol number
    symbol = symbols[symbolIdx]  # ['DU1', 'JB1', 'KE1', 'OE1', 'RX1', 'TU1', 'US1']
    print(symbol)

    symbolFolder = os.path.join(elements, symbols[symbolIdx])
    files = sorted(os.listdir(symbolFolder)) # all the files in the folder
    print(list(files))
    #
    # fileIdx = 0 # pick the first file
    # SymbolFileIdxPath = os.path.join(symbolFolder, files[fileIdx])
    # symbolFileDict = open_pickle_filepath(SymbolFileIdxPath) # loading the selected file
    #
    #
    # barCh = bars[2]
    # date = list(symbolFileDict[barCh].keys())[0]
    # print(date)
    # print(barCh)
    # print(symbolFileDict[barCh])
    # #
    # # df = symbolFileDict[barCh][date]
    # # tick_bar_df = symbolFileDict['tick_bar'][date]
    # # volume_bar_df = symbolFileDict['volume_bar'][date]
    # # usd_volume_bar_df = symbolFileDict['usd_volume_bar'][date]
    # # calendar_bar_df = symbolFileDict['calendar_bar'][date]
    # # #
    # # print(calendar_bar_df)
