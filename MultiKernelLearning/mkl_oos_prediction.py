from collections import defaultdict

import sys

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import mkl_data_processing as mkldp
import pickle

import pandas as pd
import numpy as np
import multiprocessing

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy import generators
from sklearn.svm import SVC
import os
from sklearn.metrics import accuracy_score
import torch
from MKLpy.algorithms import AverageMKL, EasyMKL  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM
from MKLpy.metrics import pairwise
from MKLpy.utils.misc import identity_kernel
from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures

def open_pickle_filepath(pickle_file):
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file


def forward_Dates(list_of_keys, current_date):
    """
    return all the forward looking dates for each idxKey we use for training

    :param list_of_keys: dates i have model dates for out of sample
    :param current_date: current model date
    :return: forward dates for applying the fitted model
    """
    lookAheadKeys = sorted(i for i in list_of_keys if i > current_date)
    return lookAheadKeys


if __name__ == '__main__':
    mainPath = mkldp.paths('main')

    # symbols = sorted(os.listdir(mkldp.paths('symbols_features')))
    jointFeatureLocation = os.path.join(mainPath, "ExperimentCommonLocs/JointLocationsDicts")


    mklOOSPredictionPath = os.path.join(mainPath, "ExperimentCommonLocs/MKLOOSPredictions")
    allFiles = sorted(os.listdir(jointFeatureLocation))
    #allFiles = [f for f in allFilesTOTAL if str("DGE") in f] # ex CNA
    # needs to be parallelised from here

    #for file in allFiles:  # pick a file- any file
    #this reads the joint file location where everything is stored

    def parallelised_OOS_function(file):

        select_file_path = os.path.join(jointFeatureLocation, file)  # formulate the path


        symbol = file.split("_")[0]
        print('Symbol:----->', symbol)
        select_label_idx = select_file_path.split("_")[9]  # pull out the label _idx

        unpickled_select_file = open_pickle_filepath(select_file_path)  # unplickle the select file

        hmm_keys = sorted(list(unpickled_select_file.keys()))  # hmm keys for the select file.

        for hmm_date_key in hmm_keys:  # pick and hmm date
            feature_label_keys = sorted(
                unpickled_select_file[hmm_date_key].keys())  # each key here unlocks a feature and label set

            for feature_label_date in feature_label_keys:  # make a list of all the feature dates
                print(feature_label_date)

                # fit in sample model here

                # this is the feature path
                features_file_path = unpickled_select_file[hmm_date_key][feature_label_date][0]

                # this is the labels path

                labels_file_path = unpickled_select_file[hmm_date_key][feature_label_date][1]
                # --> enter an if statement in case the labels do not exist !

                if os.path.isfile(labels_file_path):

                    print('ok----->have feature date:', feature_label_date)  # if you got to this point we have data so we can mov eon
                    labels = pd.read_csv(labels_file_path)  # open labels file
                    label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                    features = open_pickle_filepath(features_file_path)  # opens features file
                    hmm_features = nfu.hmm_features_df(features)  # get the hmm features out, so unpack the tuples!
                    print('loaded features and labels ')
                    if hmm_features.isnull().values.all():  # checking that the HMM features are actually not null
                        print('hmm shape null')
                        continue
                    else:  # if features not null then start moving on!
                        forward_dates_list = forward_Dates(list_of_keys=feature_label_keys,
                                                           current_date=feature_label_date)
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

                        if df_final.shape[0] < 10:  # make sure it all looks reasonable
                            print(' the ratio of classes is too low. try another label permutation')
                            continue
                        else:

                            print("starting model fit")
                            # put the features in a tensor format
                            X = np.asarray(df_final.values)  # need this for torch
                            Xtr = normalization(rescale_01(torch.Tensor(X)))  # features in a tensor format

                            Ytr = torch.Tensor(y_train.values)  # put the labels in a tensor format
                            print('-----------------next bit is the in -sample Tr Model ------------------')
                            try:
                                # in sample kernel
                                base_learner = SVC(C=0.1)
                                KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(1, 11)] + [
                                    identity_kernel(len(Ytr))]
                                # this can take the value C=0.1 or 1 based on what we
                                # have trained
                                clf = EasyMKL(lam=0.2, multiclass_strategy='ova', learner=base_learner).fit(KLtr, Ytr)
                                # lam = 0.2 based on cross-validation

                                mkl_avg = AverageMKL().fit(KLtr, Ytr)


                                print('done with kernel')

                                # --- OUT OF SAMPLE HERE --- ----------------------- ------------------------ ------------------ #
                                for forward_date in forward_dates_list:
                                    print('Doing Forward Date:', forward_date)

                                    forward_features_file_path = unpickled_select_file[hmm_date_key][forward_date][
                                        0]  # this is the feature path

                                    forward_labels_file_path = unpickled_select_file[hmm_date_key][forward_date][
                                        1]  # this is the labels path

                                    if os.path.isfile(forward_labels_file_path) and os.path.isfile(
                                            forward_features_file_path):  # if label file exists I can traing
                                        print('ok----->',
                                              forward_features_file_path)  # if you got to this point we have data so we can mov eon

                                        labels = pd.read_csv(forward_labels_file_path)  # open labels file for a forward date

                                        label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])

                                        features = open_pickle_filepath(
                                            forward_features_file_path)  # opens features file- so now you have a forward date sorted out

                                        hmm_features = nfu.hmm_features_df(
                                            features)  # get the hmm features out, so unpack the tuples!
                                        print('loaded FORWARD features and labels ')

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
                                            y_te = df.reindex(columns=df.columns[df.columns.str.contains(pat='label')]) # training labels
                                            if df_final.shape[0] < 10:  # make sure it all looks reasonable
                                                print(' the ratio of classes is too low. try another label permutation')
                                                continue
                                            else:

                                                print("starting model fit")
                                                # put the features in a tensor format
                                                X = np.asarray(df_final.values)  # need this for torch
                                                Xte = normalization(rescale_01(
                                                    torch.Tensor(X)))  # features in a tensor format - but this now becomes an Xte

                                                Yte = torch.Tensor(y_te.values.ravel())  # put the labels in a tensor format
                                                print('-----------------first bit done------------------')
                                                try:
                                                    print('----------------- about to do OOS ------------------')
                                                    #KLte= generators.RBF_generator(Xte, Xtr, gamma=[.01, .1, .25,.5])
                                                    KLte = [pairwise.homogeneous_polynomial_kernel(Xte, Xtr, degree=d) for d in range(1, 11)]
                                                    y_pred = clf.predict(KLte) # predictions
                                                    y_score = clf.decision_function(KLte)  # rank
                                                    print(' and the score', y_score)
                                                    accuracy = accuracy_score(Yte, y_pred)
                                                    print('Accuracy Score: %.3f' % (accuracy))
                                                    y_preds_average = mkl_avg.predict(KLte)  # predict the output class
                                                    oos_mkl_results = defaultdict(dict)
                                                    oos_mkl_results['MKL_POLY_OVA'][forward_date]= mkldp.evaluate_predictions(Yte, y_pred)
                                                    oos_mkl_results['AVG'][forward_date] = mkldp.evaluate_predictions(Yte, y_preds_average)

                                                    pickle_out_filename = os.path.join('/media/ak/DataOnly/',
                                                                                       "ExperimentCommonLocs/MKLOOSPredictions/",
                                                                                       "_".join(
                                                                                           (symbol, 'oos_date',
                                                                                            str(forward_date), 'label_idx', select_label_idx ,
                                                                                            str(select_label_idx), 'fit_date:',
                                                                                            str(feature_label_date ), 'POLYAVG',
                                                                                            'MKLOOSResults.pkl')))
                                                    print('about to save to... ', pickle_out_filename)
                                                    pickle_out = open(pickle_out_filename, 'wb')
                                                    pickle.dump(oos_mkl_results, pickle_out)
                                                    pickle_out.close()

                                                except (ValueError, TypeError, EOFError, IndexError):
                                                    continue
                            except (ValueError, TypeError, EOFError, IndexError):
                                continue
                else:
                    continue


    with multiprocessing.Pool(processes=8) as process_pool:
        process_pool.map(parallelised_OOS_function, allFiles)
