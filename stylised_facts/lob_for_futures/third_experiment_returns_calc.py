import glob
import fathon
import pandas as pd
from fathon import fathonUtils as fu
import numpy as np
import time
import itertools
import pickle
import os
from collections import defaultdict
import lob_for_futures as lobfut

# prepping for fathon experiments again
laptop_OS_folder = '/media/ak/OS/Data/FuturesDataSemiProcessed'
symbols = sorted(os.listdir(laptop_OS_folder))


def symbolFilePath(symbolPath, fileIdx):
    return os.path.join(symbolPath, os.listdir(symbolPath)[fileIdx])


# params i need for fathon
winSizes = fu.linRangeByStep(5, 50)
revSeg = True
qs = np.arange(-3, 4, 0.1)
polOrd = 3

bars = ['tick', 'volume', 'calendar', 'dollar']

bar_returns = dict()
standarised_returns = dict()
# -storage locations! -
# mfdfa_n_F_dict = defaultdict(dict)
# mfdfa_H_dict = defaultdict(dict)
# mfdfa_tau_dict = defaultdict(dict)
# mfdfa_alpha_spect = defaultdict(dict)
# dict_dfa = defaultdict(dict)
# h_dict = defaultdict(dict)
laptop_OS_folder = '/media/ak/OS/Data/FuturesDataSemiProcessed'
LaCie_ProcessedData = '/media/ak/LaCie/ProcessedSampledData/'
output_location = '/media/ak/T7/August11th2022Experiments'
# erase the below if not needed
# tick_dict = dict()
# usd_dict = dict()
# calendar_dict = dict()
# volm_dict = dict()
if __name__ == '__main__':
    # go through all the dataframes and take out n, f and h, h_ intercept
    # somehow one list comprehension blows up in memory
    # symbolIdx = 5

    symbolIdx = 2
    # this is the iterable number across all the symbols in the folder

    symbol = symbols[symbolIdx]  # show symbol -redundant
    symbolPath = os.path.join(laptop_OS_folder, symbol)  # this is the necessary location
    print(symbol)

    # pklFileIdx = 10
    # so now pick a pickle file, this creates a nested loop type situation
    for pklFileIdx in range(0, 100):
        try:
            loaded_dd = pd.read_pickle(symbolFilePath(symbolPath, pklFileIdx))
            date_hack = list(loaded_dd.keys())[0]  # get the date out hack
            bar_returns['dollar'] = (
                lobfut.returns(lobfut.apply_micro_structure_features(loaded_dd[date_hack]['dollar']).micro_price)).replace([np.inf, -np.inf], 0)
            bar_returns['tick'] = (
                lobfut.returns(lobfut.apply_micro_structure_features(loaded_dd[date_hack]['tick']).micro_price)).replace([np.inf, -np.inf], 0)
            bar_returns['volume'] = (
                lobfut.returns(lobfut.apply_micro_structure_features(loaded_dd[date_hack]['volume']).micro_price)).replace([np.inf, -np.inf], 0)
            bar_returns['calendar'] = (
                lobfut.returns(lobfut.apply_micro_structure_features(loaded_dd[date_hack]['calendar']).micro_price)).replace([np.inf, -np.inf], 0)
            # # print(bar_returns['dollar'] - bar_returns['dollar'].values.mean())
            pickle_out_filename = os.path.join(output_location,'Returns', symbol, date_hack + '.pkl')
            pickle.dump(bar_returns, open(pickle_out_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            print('just saved: ', pickle_out_filename)
        except EOFError:
            continue