import sys
sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import jsonpickle
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.metrics import pairwise
# from MKLpy.metrics import margin, radius, ratio, trace, frobenius
# from MKLpy.preprocessing import kernel_normalization
from MKLpy import generators
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import os
import torch
import time
import new_alternate_single_svm as nalsvm


# just pick symbols I have joint locations
jointLocsSymbols = list(np.unique([f.split("_")[0] for f in os.listdir(nalsvm.jointLocationsPickleFolder)]))

if __name__ == '__main__':
    symbol = 'CCL.L'
    good_list= list()

    if symbol in jointLocsSymbols:
        print('ok to go')
        # alternate_label_idx = 1
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
            nalsvm.gc.collect()  # continue
            for joint_key_idx, joint_key_date in enumerate(joint_keys):
                nalsvm.gc.collect()
                # this is a date - and we will enumerate through the keys
                # getting features and labels
                # nalsvm.logmemoryusage("Before feature creation")
                features, labels = nalsvm.ticker_features_labels(jointLocationsDictionary[joint_keys[joint_key_idx]])
                print('Joint Key Date:',joint_key_date)
                label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                features_df = nalsvm.hmm_features_df(features)
                nalsvm.gc.collect()
                # features data-frame - this just unbundles the features into a dataframe
                # lets get all the features in order now#
                market_features_df = nalsvm.CreateMarketFeatures(nalsvm.CreateMarketFeatures(
                    nalsvm.CreateMarketFeatures(df=nalsvm.CreateMarketFeatures(
                        df=labels).ma_spread_duration()).ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
                nalsvm.gc.collect()
                df_concat = pd.DataFrame(pd.concat([features_df, market_features_df], axis=1, sort='False').dropna())
                df = df_concat[df_concat[label_name].notna()]
                df_final = df.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                            'Volume', label_name])
                y_labels_train = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]
                if df_final.shape[0] < 3:
                    print('problem')
                else:
                    print('-----------no problem-----------')
                    good_list.append(joint_key_date)
                    Ytr = torch.tensor(
                        df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0].values)
                    df_final = torch.Tensor(df_final.values)
                    start =time.time()
                    Xtr = normalization(rescale_01(df_final)) # feature scaling in [0,1]
                    print('first bit done')
                    nalsvm.gc.collect()
                    KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(11)]
                    print('done')
                    base_learner = SVC(C=0.1)
                    try:
                        clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr, Ytr)
                        nalsvm.logmemoryusage("Before garbage collect")
                        nalsvm.gc.collect()
                        print('done too')
                        end = time.time()
                        print(f'it took {end - start} seconds!')
                    except ValueError:
                        continue

