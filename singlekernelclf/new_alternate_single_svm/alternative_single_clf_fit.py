from fileutils import DataLoader as DataLoader
from fileutils import paths
from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures
from clfutils import FitModels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, hamming_loss
import os
import multiprocessing
import psutil
import pickle
from collections import defaultdict
import itertools


def logmemoryusage(msg):
    # function to log memory usage
    process = psutil.Process(os.getpid())
    print('Memory usage at %s is %smb.' % (msg, process.memory_info().rss / 1000 / 1000))


def open_pickle_filepath(pickle_file):
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file


if __name__ == '__main__':
    symbols = sorted(os.listdir(paths('symbols_features')))

    alternate_labels_nos = [1, 2, 3, 4, 5, 6, 7]  # we have 7 alternative data types
    mainPath = paths('main')

    def parallised_function(symbol, label_idx):

        symbolData = DataLoader(mainPath, symbol)
        # hmm_dates_list = (symbolData.hmm_dates_list) hmm_dates_models_list

        for hmm_date_idx, hmm_date in enumerate(sorted(symbolData.hmm_dates_list)):
            hmm_features_date_path = os.path.join(symbolData.symbol_features_path, hmm_date)
            symbol_feature_paths = {feature_file.split("_")[5]: os.path.join(hmm_features_date_path, feature_file) for
                                    f, feature_file in enumerate(sorted(os.listdir(hmm_features_date_path)))}
            keys = sorted(list(symbol_feature_paths.keys()))
            for key in keys:  # <--- this is the label key in older versions of the code!
                labels_file_path = os.path.join(symbolData.symbol_specific_label_path(label_idx), key + ".csv")
                # this is the label path in older versions of the code
                best_svc_dict = defaultdict(dict)
                if os.path.isfile(labels_file_path):  # check that this is a real path
                    print("can train")
                    print(" reading labels")  # this is the labels path!
                    labels = pd.read_csv(labels_file_path)
                    label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                    print(symbol_feature_paths[key])  # this is the labels path! in the correct order !!!!
                    logmemoryusage("Before garbage collect")
                    hmm_features = nfu.hmm_features_df(open_pickle_filepath(symbol_feature_paths[key]))
                    if hmm_features.isnull().values.all():  # checking that the HMM features are actually not null
                        continue
                    else:  # if features not null then start moving on!
                        market_features_df = CreateMarketFeatures(
                            CreateMarketFeatures(
                                CreateMarketFeatures(df=CreateMarketFeatures(df=labels).ma_spread_duration())
                                    .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe

                        df_concat = pd.DataFrame(
                            pd.concat([hmm_features, market_features_df], axis=1, sort='False').dropna())

                        df = df_concat[df_concat[label_name].notna()]
                        df_final = df.drop(
                            columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', 'Volume',
                                     label_name])
                        y_train = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]  # training labels
                        if df_final.shape[0] < 10:
                            print(' the ratio of classes is too low. try another label permutation')
                            continue
                        else:
                            try:
                                print("starting model fit")
                                X_train = MinMaxScaler().fit_transform(df_final)
                                models_cls = FitModels(X_train, y_train)
                                print(models_cls.best_svc_clf())
                                best_svc_dict[symbol][key] = {'SVC': models_cls.best_svc_clf()}
                                # this is a symbol, label-date fitted SVC

                            except ValueError:
                                continue
                                logmemoryusage("at the end")

                else:
                    print("#################### Your Labels File does not exist ----- ####")
                    continue
                logmemoryusage("Before garbage collect")

            # create features - first HMM and second some Market Features!

            pickle_out_filename = os.path.join(mainPath, "ExperimentCommonLocs/FittedModels", "_".join(
                (symbol, 'model_fit_date', str(key), str(alternate_labels_nos[label_idx]), 'SingleKernelSVC.pkl')))
            pickle_out = open(pickle_out_filename, 'wb')
            pickle.dump(best_svc_dict, pickle_out)
            pickle_out.close()


    with multiprocessing.Pool(processes=18) as process_pool:
        process_pool.starmap(parallised_function, itertools.product(symbols, alternate_labels_nos))
