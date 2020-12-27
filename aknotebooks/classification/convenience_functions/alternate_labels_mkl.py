from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import fileutils as fileutils
import gc
import psutil
import pickle
import fnmatch
import os
import time
import clfutils
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.model_selection import cross_val_score, cross_val_predict
from MKLpy.metrics import pairwise
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, hamming_loss
import new_alternate_single_svm as nalsvm
import altlabels_single_kernel_oos as skoos

if __name__ == "__main__":
    symbol = 'AAL.L'
    jointLocsSymbols = list(np.unique([f.split("_")[0] for f in os.listdir(nalsvm.jointLocationsPickleFolder)]))
    print(jointLocsSymbols)
    for symbol in jointLocsSymbols:
        print('ok to go')
        for alternate_label_idx in range(0, 4):
            symbol_idx = jointLocsSymbols.index(symbol)  # dont particularly need this!
            print(symbol, ' and labels ', nalsvm.labels_pickle_files[alternate_label_idx])
            data_cls = nalsvm.AlternateLabelFeaturesLoader(path_main=nalsvm.dataDrive,
                                                           symbol=jointLocsSymbols[symbol_idx],
                                                           alternate_label_idx=alternate_label_idx,
                                                           jointLocationsPickleInput=nalsvm.jointLocationsPickleFolder)
            jointLocationsDictionary = (data_cls.load_pickled_in_filename(data_cls.return_pickled_dict()))
            joint_keys = data_cls.joint_loc_pickle_keys(data_cls.return_pickled_dict())
            nalsvm.logmemoryusage("Before garbage collect")
            gc.collect()  # continue
            count_i = 0
            for joint_key_idx, joint_key_date in enumerate(joint_keys):
                # this is a date - and we will enumerate through the keys
                # getting features and labels
                nalsvm.logmemoryusage("Before feature creation")
                features, labels = nalsvm.ticker_features_labels(jointLocationsDictionary[joint_keys[joint_key_idx]])
                print(joint_key_date)
                label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                features_df = nalsvm.hmm_features_df(
                    features)  # features data-frame - this just unbundles the features into a dataframe
                # lets get all the features in order now#
                market_features_df = nalsvm.CreateMarketFeatures(
                    nalsvm.CreateMarketFeatures(
                        nalsvm.CreateMarketFeatures(df=nalsvm.CreateMarketFeatures(df=labels).ma_spread_duration())
                        .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
                df_concat = pd.DataFrame(pd.concat([features_df, market_features_df], axis=1, sort='False').dropna())
                df = df_concat[df_concat[label_name].notna()]
                df_final = df.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                            'Volume', label_name])
                y_labels_train = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]
                if df_final.shape[0] < 10:
                    print(' the ratio of classes is too low. try another label permutation')
                    continue
                else:
                    try:
                        X_train = MinMaxScaler().fit_transform(df_final)
                        nalsvm.logmemoryusage("After feature creation")
                        if X_train.shape[0] == y_labels_train.shape[0]:
                            nalsvm.logmemoryusage("Before starting training")
                            print('Shapes Match- starting training ')
                            # polynomial Kernels ##
                            try:
                                KLtr = [pairwise.homogeneous_polynomial_kernel(X_train, degree=d) for d in range(4)]
                                # KLte = [pairwise.homogeneous_polynomial_kernel(Xte, Xtr, degree=d) for d in range(4)]
                                print('done')
                                clf = AverageMKL().fit(KLtr, y_labels_train)  # a wrapper for averaging kernels
                                # print(clf.weights)  # print the weights of the combination of base kernels
                                print('training EasyMKL...for polynomials and RBF')
                                clfEasy = EasyMKL(lam=0.1).fit(KLtr,
                                                               y_labels_train)  # combining kernels with the EasyMKL algorithm
                                print('------')
                                print('finished training')
                                # somewhere here you need to do out of sample testing and then store all that
                                symbolForwardDates = data_cls.forwardDates(joint_keys, joint_keys[joint_key_idx])
                                oos_svc_predictions = defaultdict(dict)
                                # alias to store the data : symbol, joint Date, Label Used
                                results_predict_alias = "_".join((symbol, joint_keys[joint_key_idx, nalsvm.labels_pickle_files[alternate_label_idx]))
                                for forward_date_idx, forward_date in enumerate(symbolForwardDates):
                                    features_oos, labels_oos = nalsvm.ticker_features_labels(nalsvm.jointLocationsDictionary[symbolForwardDates[forward_date_idx]])
                                if nalsvm.hmm_features_df(features_oos).isnull().values.all():
                                    print('Problem')
                                    ## need to get all the data out for
                                    KLte = [pairwise.homogeneous_polynomial_kernel(Xte, X_train, degree=d) for d in range(4)]
                                print('done')

                            except:
                                count_i += 1
                                print(count_i)
                                print("hin failed here!")

                                continue
                    except:
                        print('Shapes dont match.')
                        pass
