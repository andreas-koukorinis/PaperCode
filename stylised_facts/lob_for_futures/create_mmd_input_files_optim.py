import os
import pandas as pd
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/directory/tothe/handshakefile/')
sys.path.append('/home/ak/Documents/PaperCode/stylised_facts')

import stylised_facts_data_utilities as sfd_utils
import lob_for_futures as lobfut
from sklearn.preprocessing import MinMaxScaler, StandardScaler

expInputFiles = '/media/ak/T71/August11th2022Experiments/ExperimentOne'
mfdfaDataFrames = '/media/ak/T71/August11th2022Experiments/mfdfaDataFrames'  # Storage of Input Files


def get_filtered_files(path, bar_choice):
    return sorted([f for f in os.listdir(path) if str(bar_choice) in f])


def read_pickle_file(file_path, bar_choice, variable):
    return pd.read_pickle(file_path)[str(bar_choice)][str(variable)]


def create_dataframe_from_dict(unpickled_all_values):
    return pd.DataFrame.from_dict(unpickled_all_values).fillna(0)


def save_dataframe_to_pickle(df, save_path, protocol=pickle.HIGHEST_PROTOCOL):
    with open(save_path, 'wb') as f:
        pickle.dump(df, f, protocol=protocol)
    print(f'saving: {save_path}')


def process_variable(variable, bar_choice, files, expInputFiles, mfdfaDataFrames, symbol):
    unpickled_all_values = dict()

    for idx, file in enumerate(files):
        file_loc = os.path.join(expInputFiles, file)
        unpickled_all_values[idx] = read_pickle_file(file_loc, bar_choice, variable)

    df = create_dataframe_from_dict(unpickled_all_values)
    output_file_name = "_".join((symbol, bar_choice, variable)) + ".pkl"
    save_loc = os.path.join(mfdfaDataFrames, output_file_name)
    save_dataframe_to_pickle(df, save_loc)


def main( symbol = 'JB1',bar_choice = 'calendar'):

    files = get_filtered_files(expInputFiles, bar_choice)

    variables = ['n_F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect']

    with ProcessPoolExecutor() as executor:
        for variable in variables:
            executor.submit(process_variable, variable, bar_choice, files, expInputFiles, mfdfaDataFrames, symbol)


if __name__ == '__main__':
    main(symbol = 'JB1',bar_choice = 'calendar')
