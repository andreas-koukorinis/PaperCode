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


    def parallised_function(symbol, label_idx):
        symbolData = DataLoader(mainPath, symbol)

        for hmm_date_idx, _ in enumerate(symbolData.hmm_dates_list):
            hmm_date = symbolData.hmm_dates_list[hmm_date_idx]
            logmemoryusage("Before garbage collect")
            features_paths = symbolData.hmm_model_date_feature_list_filepaths(hmm_date)[1]
            labels_paths = symbolData.hmm_model_feature_corrsp_labels_files(hmm_date, alternate_labels_nos[label_idx])
            logmemoryusage("Before garbage collect")
            best_svc_dict = defaultdict(dict)
            for key in list(labels_paths.keys()):
                # this i can probably parallelise- all the code her
                print(key)
                labels = pd.read_csv(labels_paths[key])
                label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                # create features - first HMM and second some Market Features!
                hmm_features = nfu.hmm_features_df(open_pickle_filepath(features_paths[key]))

                market_features_df = CreateMarketFeatures(
                    CreateMarketFeatures(CreateMarketFeatures(df=CreateMarketFeatures(df=labels).ma_spread_duration())
                                         .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe

                df_concat = pd.DataFrame(pd.concat([hmm_features, market_features_df], axis=1, sort='False').dropna())

                df = df_concat[df_concat[label_name].notna()]
                df_final = df.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                            'Volume', label_name])

                y_train = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]

                if df_final.shape[0] < 10:
                    print(' the ratio of classes is too low. try another label permutation')
                    continue
                else:
                    try:
                        X_train = MinMaxScaler().fit_transform(df_final)
                        models_cls = FitModels(X_train, y_train)
                        print(models_cls.best_svc_clf())
                        best_svc_dict[symbol][key] = {'SVC': models_cls.best_svc_clf()}
                        # this is a symbol, label-date fitted SVC

                    except ValueError:
                        continue
                        logmemoryusage("at the end")

                pickle_out_filename = os.path.join(mainPath,"ExperimentCommonLocs/FittedModels", "_".join(
                    (symbol,str(key), str(alternate_labels_nos[label_idx]), 'SingleKernelSVC.pkl')))
                pickle_out = open(pickle_out_filename, 'wb')
                pickle.dump(best_svc_dict, pickle_out)
                pickle_out.close()

    symbols = os.listdir(paths('symbols_features'))
    print(symbols)

    alternate_labels_nos = [1, 2, 3, 4, 5, 6, 7]  # we have 7 alternative data types
    mainPath = paths('main')

    with multiprocessing.Pool(processes=4) as process_pool:
        process_pool.starmap(parallised_function, itertools.product(symbols, alternate_labels_nos))
