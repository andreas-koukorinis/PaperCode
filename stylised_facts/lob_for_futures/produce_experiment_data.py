import pandas as pd
import numpy as np
import sys
import multiprocessing
import time

sys.path.append(('/home/ak/Research/PaperCode/stylised_facts'))
sys.path.append('/home/ak/Research/PaperCode/stylised_facts')
# import stylised_facts_data_utilities as sfd_utils
import lob_for_futures as lobfut
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from multiprocessing import Pool, freeze_support
import pickle
from multiprocessing import Pool

scaler = MinMaxScaler()
standard_scaler = StandardScaler()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sys.path.insert(0, '/directory/tothe/handshakefile/')
sys.path.append('/home/ak/Documents/PaperCode/stylised_facts')
## data files

#laptop_OS_folder = '/media/ak/T71/FuturesDataSemiProcessed'

laptop_OS_folder = '/media/ak/T71/FuturesDataSemiProcessed'
LaCie_ProcessedData = '/media/ak/LaCie/ProcessedSampledData/'
# returns_data = '/media/ak/T7/August11th2022Experiments/Returns/'
t7 = '/media/ak/T71/'
t7_folder = os.path.join(t7, 'FuturesDataSemiProcessed')
# june_ext = os.path.join(t7, 'June4th2022Experiments')
# returns_data = [f for f in os.listdir(june_ext) if '_returns' in f]
experimentsLocation = '/media/ak/T71/August11th2022Experiments/ExperimentInputFiles/'
# here i start with RX1 to do all the experiments in one go
symbols = os.listdir(laptop_OS_folder)


# symbol_test_folder = os.path.join(laptop_OS_folder, symbols[0]) rx_folder = os.path.join(laptop_OS_folder,
# 'RX1') du_folder = os.path.join(t7_folder,  'DU1')
# , 'FB1') make this a bit more dynamic to take any function in here files = os.listdir(du_folder)


def produce_experiment_data(chosen_df):
    """
    # this is somewhat unit-tested in the August 7th 2022 notebook
    # function to produce dataframes for experiments
    # input: chosen df - this is a dataframe that we apply the microstructure features and the vol estimation features
    # output: experiment - dataframe with the features we want experiments for
    # written in August 2022
    # re-write it as part of the _init_ file

    """
    with np.errstate(divide='ignore'):
        chosen_df_micro = lobfut.apply_micro_structure_features(chosen_df)  # get micro structure df
        vol_class = lobfut.volatilityEstimation(chosen_df)  # get the vol class

        # features I need: micro_price changes / vols /skews /etc:

        experiment_df = chosen_df_micro.loc[:, ['micro_price', 'price_imbalance',
                                                'pct_change_micro_price', 'weighted_activity_spread', ]]
        try:
            experiment_df['GK_vol'] = pd.Series(
                list(vol_class.garmanKlass(5)))  # get a sample of Garman - Klass resampled for 5 clicks
            experiment_df['arrival_rates'] = pd.Series(
                vol_class.arrival_rates().reshape(vol_class.arrival_rates().shape[0], )).replace([np.inf, -np.inf],
                                                                                                 0).fillna(0)
            X = experiment_df.pct_change_micro_price.replace([np.inf, -np.inf], 0).values.reshape(-1, 1)
            norm_scaler = StandardScaler().fit(X)  # normalised scaling by mean and std
            min_max_scaler = MinMaxScaler().fit(X)  # min max scaling

            experiment_df['returns_normalised'] = norm_scaler.transform(
                X)  # use this format to get rid of the prior issues
            experiment_df['returns_mix_max'] = min_max_scaler.transform(X)  # get both issues of
            rs, rk = vol_class.realised_skewness_kurtosis()  # get skew/kurt

            experiment_df['skew'] = pd.Series(list(rs))
            experiment_df['kurt'] = pd.Series(list(rk))

            experiment_df['median_traded_volume'] = chosen_df_micro[['total_traded_volume_open',
                                                                     'total_traded_volume_high',
                                                                     'total_traded_volume_low',
                                                                     'total_traded_volume_close']].quantile(0.5, axis=1)
            experiment_df['jumps_test'] = pd.Series(vol_class.jumps_test(rollingWindow=5, sampling_param=0))
            experiment_df['relz_var'] = pd.Series(vol_class.realised_variance(rollingWindow=5))
            experiment_df['trip_quart'] = pd.Series(vol_class.tripower_quarticity(rollingWindow=5, sampling_param=0))

            experiment_df = experiment_df.replace([np.inf, -np.inf], 0).fillna(0)  # final clean up
        except (ValueError, RuntimeWarning):
            print("error")
            pass
    experiment_df = experiment_df.replace([np.inf, -np.inf], 0).fillna(0)  # final clean up

    return experiment_df


if __name__ == '__main__':
    """
    function that takes the index of a file, that index corresponds to a point in a list of files that are already
    aligned between trades and quotes and have been information clock processed
    then applies the extraction of features
    """

    # symbol = 'RX1'
    def produce_and_dump(files_idx_, symbol_, choice_bar_):
        symbol = symbol_  # and this
        symbol_folder_path = os.path.join(t7_folder, str(symbol))
        all_files = os.listdir(symbol_folder_path)
        files = [f for f in all_files if str('Returns_') not in f]

        choice_bar = choice_bar_  # change this
        date_idx = files[files_idx_].split(".")[0]
        print(date_idx)

        idx_file_path = os.path.join(symbol_folder_path,
                                     files[files_idx_])  # the input here needs to be dynamic not du_folder or rx_folder
        choice_df = pd.read_pickle(idx_file_path)[date_idx][choice_bar]
        exp_df = produce_experiment_data(choice_df)
        pickle_out_returns = os.path.join(experimentsLocation,symbol, "".join(
            (str(symbol) + "_" + str(choice_bar) + "_" + str(date_idx) + "_exp_df.pkl")))
        pickle.dump(exp_df, open(pickle_out_returns, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print('saved:', pickle_out_returns)


    symbol = 'FB1'  # and this --> this symbol needs to be put in another folder as well
    symbol_folder_path = os.path.join(t7_folder, str(symbol))
    files = os.listdir(symbol_folder_path)
    pool = Pool(6)
    start_time = time.perf_counter()
    # on processes change this to take the input of a symbol above!
    processes = [pool.apply_async(produce_and_dump, args=(files_idx_,str(symbol),'dollar')) for files_idx_ in range(0, len(files))]
    # can change this range if need be to complete the data easier?
    # maybe make this a bit more modular?
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    print(result)
