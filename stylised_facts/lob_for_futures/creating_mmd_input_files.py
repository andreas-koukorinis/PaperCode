import os
import pandas as pd
import numpy as np
import multiprocessing
import time
import sys

sys.path.insert(0, '/directory/tothe/handshakefile/')
sys.path.append('/home/ak/Documents/PaperCode/stylised_facts')
import stylised_facts_data_utilities as sfd_utils
import lob_for_futures as lobfut
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from multiprocessing import Pool, freeze_support
import pickle
import pandas as pd
import numpy as np

expInputFiles = '/media/ak/T71/August11th2022Experiments/ExperimentOne'
mfdfaDataFrames = '/media/ak/T71/August11th2022Experiments/mfdfaDataFrames'  # Storage of Input Files

if __name__ == '__main__':
    bar_choice = 'calendar'
    # [volume, tick, dollar, calendar]
    symbol = 'JB1'
    files = sorted([f for f in os.listdir(expInputFiles) if str(bar_choice) in f])
    # format: SYMBOl_dollar_mfSpect

    unpickled_all_values = dict() # intermediary storage
    variable = 'tau'
    variables = ['n_F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect']
    # n f is kind of problematic, so we need a condition in there
    for variable in variables:
        for idx, file in enumerate(files):
            file_loc = os.path.join(expInputFiles, file)
            unpickled_all_values[idx] = pd.read_pickle(file_loc)[str(bar_choice)][str(variable)]
        df = pd.DataFrame.from_dict(unpickled_all_values).fillna(0)
        output_file_name = "_".join((symbol, bar_choice, variable)) + ".pkl"
        save_loc = os.path.join(mfdfaDataFrames, output_file_name)
        pickle.dump(df, open(save_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print('saving:',save_loc )