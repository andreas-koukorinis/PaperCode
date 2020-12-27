import sys

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import pickle
import pandas as pd
import numpy as np
import os
import time
import new_alternate_single_svm as nalsvm

# file to do the data processing of clean data so we can fit models quite easily and reduce memory
# just pick symbols I have joint locations
jointLocsSymbols = list(np.unique([f.split("_")[0] for f in os.listdir(nalsvm.jointLocationsPickleFolder)]))


def storage_location(symbol):
    storage_location = os.path.join(nalsvm.dataDrive, 'JointLocationsAlternateDataClean', symbol)
    return storage_location


def oos_results_location(symbol):
    oos_results_location = os.path.join(nalsvm.dataDrive, 'JointLocationsAlternateDataClean', 'OOS_Results', symbol)
    if not os.path.exists(oos_results_location):
        os.makedirs(oos_results_location)
    return oos_results_location


def cross_validation_results_location(symbol):
    storage_location = os.path.join(nalsvm.dataDrive, 'JointLocationsAlternateDataClean', 'CV_Results', symbol)
    if not os.path.exists(storage_location):
        os.makedirs(storage_location)
    return storage_location


if __name__ == '__main__':
    symbol = 'CPG.L'
    storage_location = storage_location(symbol)
    if not os.path.exists(storage_location):
        os.makedirs(storage_location)
    good_list_dict = dict()

    if symbol in jointLocsSymbols:
        for alternate_label_idx in range(0, 4):
            print(symbol, ' and labels ', nalsvm.labels_pickle_files[alternate_label_idx])
            data_cls = nalsvm.AlternateLabelFeaturesLoader(path_main=nalsvm.dataDrive,
                                                           symbol=jointLocsSymbols[jointLocsSymbols.index(symbol)],
                                                           alternate_label_idx=alternate_label_idx,
                                                           jointLocationsPickleInput=nalsvm.jointLocationsPickleFolder)
            jointLocationsDictionary = (data_cls.load_pickled_in_filename(data_cls.return_pickled_dict()))
            joint_keys = data_cls.joint_loc_pickle_keys(data_cls.return_pickled_dict())
            nalsvm.logmemoryusage("Before garbage collect")
            nalsvm.gc.collect()  # continue
            for joint_key_idx, joint_key_date in enumerate(joint_keys):
                nalsvm.gc.collect()
                # this is a date - and we will enumerate through the keys
                # getting features and labels
                # nalsvm.logmemoryusage("Before feature creation")
                features, labels = nalsvm.ticker_features_labels(jointLocationsDictionary[joint_keys[joint_key_idx]])
                print('Joint Key Date:', joint_key_date)
                label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                features_df = nalsvm.hmm_features_df(features)
                nalsvm.gc.collect()
                # features data-frame - this just unbundles the features into a dataframe
                # lets get all the features in order now#
                market_features_df = nalsvm.CreateMarketFeatures(nalsvm.CreateMarketFeatures(
                    nalsvm.CreateMarketFeatures(df=nalsvm.CreateMarketFeatures(
                        df=labels).ma_spread_duration()).ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
                nalsvm.gc.collect()
                # cleaned up prior versions of the code so less memory is used here. dont need all these dataframes!
                df_concat = pd.DataFrame(pd.concat([features_df, market_features_df], axis=1, sort='False').dropna())
                Xtr = df_concat.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                              'Volume', label_name])
                Ytr = df_concat[df_concat[label_name].notna()][df_concat[df_concat[label_name].notna()].columns[
                    df_concat[df_concat[label_name].notna()].columns.str.contains(pat='label')]].iloc[:, 0]
                if Xtr.shape[0] < 3:
                    print(Xtr.shape[0])
                    print('problem')
                    nalsvm.gc.collect()
                else:
                    print('-----------no problem-----------')
                    print(Xtr.shape[0])
                    good_list_dict[joint_key_date] = [Xtr, Ytr]
                    nalsvm.gc.collect()

                pickle_out_filename = os.path.join(storage_location, "_".join(
                    (symbol, nalsvm.labels_pickle_files[alternate_label_idx], 'ProcessedData.pkl')))
                pickle_out = open(pickle_out_filename, 'wb')
                pickle.dump(good_list_dict, pickle_out)
                pickle_out.close()
                print('Data stored:', pickle_out_filename)
