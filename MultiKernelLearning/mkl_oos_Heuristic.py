import os
from collections import defaultdict

import numpy as np
import pandas as pd
import sys

from sklearn.metrics import accuracy_score

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
from fileutils import DataLoader as DataLoader
from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures

from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD, HeuristicMKL, PWMK, FHeuristic, CKA  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

import mkl_data_processing as mkldp
import mkl_model_fit_selection as fit_select
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier

import pickle


mainPath = fit_select.mainPath
mklOOSPredictionPath = fit_select.os.path.join(mainPath, "ExperimentCommonLocs/MKLOOSPredictions")
OVAPath = os.path.join(mklOOSPredictionPath,'OVA')
HeuristicPath = os.path.join(mklOOSPredictionPath, 'Heuristic')
PWMKPath = os.path.join(mklOOSPredictionPath, 'PWMK')
AverageKernel = os.path.join(mklOOSPredictionPath,'AverageKernel')
jointFeatureLocation = fit_select.os.path.join(mainPath, "ExperimentCommonLocs/JointLocationsDicts")

def open_pickle_filepath(pickle_file):
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file

def features_and_labels(df):
    market_features_df = CreateMarketFeatures(CreateMarketFeatures(CreateMarketFeatures(df=CreateMarketFeatures(df=labels).ma_spread_duration()).ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe


    df_concat = pd.DataFrame(
        pd.concat([hmm_features, market_features_df], axis=1, sort='False').dropna())

    df = df_concat[df_concat[label_name].notna()]
    df_final = df.drop(
        columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', 'Volume',
                 label_name])

    y_train = df.reindex(columns=df.columns[df.columns.str.contains(pat='label')])  # training labels

    return df_final, y_train

# paths
### paths ###

mainPath = mkldp.paths('main')

symbols = sorted(os.listdir(mkldp.paths('symbols_features')))
jointFeatureLocation = os.path.join(mainPath, "ExperimentCommonLocs/JointLocationsDicts")
commonLocs = os.path.join(mainPath, "ExperimentCommonLocs/")
# common locs contains a lot of folders, including the prediction targets and the rest

mklOOSPredictionPath = os.path.join(mainPath, "ExperimentCommonLocs/MKLOOSPredictions")
dir_loc = 'Heuristic'
mklOOSPredictionPathSpecific = HeuristicPath#os.path.join(mklOOSPredictionPath, dir_loc)
allFiles = os.listdir(jointFeatureLocation)
fittedModelsPath = os.path.join(mainPath, "ExperimentCommonLocs/FittedModels")
oosPredictionsPath = os.path.join(mainPath, "ExperimentCommonLocs/OOSPredictions")

labelsAlternateLocs = {f: os.path.join(commonLocs, f) for f in
                       os.listdir(os.path.join(mainPath, "ExperimentCommonLocs/")) if str("LabelsAlternate") in f}
labelKeys = list(labelsAlternateLocs.keys())
alternate_labels_nos = {1: "LabelsAlternateOne", 2: "LabelsAlternateTwo", 3: "LabelsAlternateThree",
                        4: "LabelsAlternateFour", 5: "LabelsAlternateFive",
                        6: "LabelsAlternateSix"}  # we have 7 alternative data types

def forward_date_features_labels(forward_df, forward_hmm_features):
    '''

    :param forward_df: this is basically forward labels file, so pd.read_csv the forward labels path
    :param forward_hmm_features: this is basically nfu.hmm_features_df(open_pickle_filepath and we need it to perform the prelim check
    :return: returns the forward date features and the labels of the forward data
    '''

    forward_market_features_df = CreateMarketFeatures(CreateMarketFeatures(CreateMarketFeatures(df=CreateMarketFeatures(df=forward_df).ma_spread_duration()).ma_spread()).chaikin_mf()).obv_calc()
    # construct the market features dataframe

    fwd_label_name = str(forward_df.columns[forward_df.columns.str.contains(pat='label')].values[0])
    # extract the label

    df_fwd_concat = pd.DataFrame(
        pd.concat([forward_hmm_features, forward_market_features_df], axis=1, sort='False').dropna())

    df_fwd = df_fwd_concat[df_fwd_concat[label_name].notna()]
    df_fwd_final = df_fwd.drop(
        columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', 'Volume',
                 fwd_label_name])

    y_fwd = df_fwd.reindex(columns=df_fwd.columns[df_fwd.columns.str.contains(pat='label')])  # training labels
    return df_fwd_final, y_fwd


if __name__ == '__main__':


    symbol_idx =41
    # 25: RBS 10:CPG 16:LAND #9:CNA

    # CCL -symbol 7 -missing: 2, 5, 6
    #
    for label_idx in [3]:

        # pick a label #do 6 after
        symbol = sorted(symbols)[symbol_idx]
        C_choice = 0.1
        lam = 0.1 # lam = 0.2 based on cross-validation

        print(symbol, symbol_idx)
        symbolData = DataLoader(mainPath, symbol)  # initiate a path where all the data should be

        labelsLocation = labelsAlternateLocs[alternate_labels_nos[label_idx]]

        labelsPath = os.path.join(labelsLocation, symbol)
        hmm_dates = sorted(symbolData.hmm_dates_list)

        for hmm_date in hmm_dates:
            print(hmm_date)

            hmm_features_date_path = os.path.join(symbolData.symbol_features_path, hmm_date)

            if os.path.isdir(hmm_features_date_path) and len(os.listdir(hmm_features_date_path))>0:
                symbol_feature_paths = {feature_file.split("_")[5]: os.path.join(hmm_features_date_path, feature_file) for
                                        f, feature_file in enumerate(sorted(os.listdir(hmm_features_date_path)))}
                keys = sorted(list(symbol_feature_paths.keys()))
                print('For hmm date: ', hmm_date, '###########')
                print(keys == fit_select.forward_Dates(hmm_dates, hmm_date) )
                for key in keys: #<- parallelisation here!

                    try:

                        labels_file_path = os.path.join(symbolData.symbol_specific_label_path(label_idx), key + ".csv")
                        print(os.path.isfile(labels_file_path))

                        if os.path.isfile(labels_file_path):  # check that this is a real path

                            print(" reading labels")  # this is the labels path!
                            labels = pd.read_csv(labels_file_path)
                            label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                            fit_select.logmemoryusage("Before garbage collect")
                            hmm_features = nfu.hmm_features_df(open_pickle_filepath(symbol_feature_paths[key]))

                            if hmm_features.isnull().values.all():  # checking that the HMM features are actually not null

                                pass

                            else:

                                print('can fit and predict!')
                                Xtr, Ytr = features_and_labels(labels)
                                if Xtr.shape[0]<10:
                                    print('ratio is too low')
                                    continue
                                else:
                                    forward_dates_list =fit_select.forward_Dates(keys, key)
                                    print('the number of forward dates is:',len(forward_dates_list))

                                    print('starting model fit')


                                    X_tr = fit_select.normalization(fit_select.rescale_01(np.array(Xtr)))
                                    Y_tr = fit_select.torch.Tensor(Ytr.values.ravel())

                                    base_learner = fit_select.SVC(C=C_choice)
                                    try:

                                        KLtr = [fit_select.pairwise.homogeneous_polynomial_kernel(X_tr, degree=d) for d in range(1, 11)] + [fit_select.identity_kernel(len(Y_tr))]
                                        # clf = fit_select.EasyMKL(lam=lam, multiclass_strategy='ova', learner=base_learner).fit(KLtr, Y_tr)
                                        #
                                        mklFHeuristic = FHeuristic().fit(KLtr, Y_tr)
                                        # mklCKA = CKA().fit(KLtr, Y_tr)

                                        y_pred = mklFHeuristic.predict(KLtr) #predictions
                                        y_score = mklFHeuristic.decision_function(KLtr) #rank
                                        accuracy = accuracy_score(Y_tr, y_pred)
                                        print ('Accuracy score: %.3f' % (accuracy))

                                        # replace the mklFheuristic w your method of choice


                                        print(' and the score', y_score)
                                        print('work on forward dates now')
                                        print('##############')
                                        oos_mkl_results = defaultdict(dict)
                                        # force memory clean of pieces of torch
                                        del Xtr
                                        del KLtr

                                        fit_select.gc.collect()


                                        for forward_date in forward_dates_list:
                                            forward_labels_file_path = os.path.join(symbolData.symbol_specific_label_path(label_idx), forward_date + ".csv")
                                            print(os.path.isfile(forward_labels_file_path))

                                            if os.path.isfile(forward_labels_file_path):  # check that this is a real path
                                                print("reading forward labels")  # this is the labels path!

                                                forward_labels = pd.read_csv(forward_labels_file_path)

                                                forward_label_name = str(forward_labels.columns[forward_labels.columns.str.contains(pat='label')].values[0]) #extracting the forward label name
                                                fit_select.logmemoryusage("Before garbage collect")

                                                forward_hmm_features = nfu.hmm_features_df(open_pickle_filepath(symbol_feature_paths[forward_date])) # estimating the features
                                                Xte, Yte = forward_date_features_labels(forward_df = forward_labels , forward_hmm_features = forward_hmm_features)

                                                if Xte.shape[0]<10:
                                                    print('..but the ratio is too low')
                                                    print(Xte)
                                                    pass
                                                else:
                                                    print('...and can do damage!')
                                                    chunk_size = int(Xte.shape[0] / 9)
                                                    for start in range(0, Xte.shape[0], chunk_size):
                                                        X_te_subset = Xte.iloc[start:start+chunk_size]
                                                        Y_te_subset = Yte.iloc[start:start+chunk_size]

                                                        X_te = fit_select.normalization(fit_select.rescale_01(np.array(X_te_subset)))
                                                        Y_te = fit_select.torch.tensor(Y_te_subset.values.ravel())

                                                        del X_te_subset
                                                        del Y_te_subset



                                                        fit_select.gc.collect()

                                                        # use the
                                                        # underscore if it has been passed to torch and normalised
                                                        try:
                                                            chunk_list = dict()

                                                            # construct the KL for the case of the testing and training sets
                                                            KLte = [fit_select.pairwise.homogeneous_polynomial_kernel(X_te,X_tr, degree=d) for d in range(1,11)]
                                                            KLte.append(fit_select.torch.zeros(KLte[0].size()))

                                                            # do the prediction using the fitted learner
                                                            Y_pred =mklFHeuristic.predict(KLte)
                                                            del KLte
                                                            fit_select.gc.collect()


                                                            accuracy = fit_select.accuracy_score(Y_te, Y_pred)
                                                            print('for forward date: ',forward_date, ' the accuracy is :', accuracy)
                                                            # now a dictionary that saves the evaluation of the prediction using
                                                            # key and forward OOS Date
                                                            chunk_list[start+chunk_size] = mkldp.evaluate_predictions(Y_te, Y_pred)
                                                            print('done chunk: ', start+chunk_size)
                                                            oos_mkl_results[key][forward_date, (start+chunk_size)] = chunk_list
                                                            oos_hash_file = "".join((str(symbol),'_',
                                                                                     'HeuristicMKL_poly_kernel','_Label_idx:_'+str(label_idx)+'_',str(key),'_',
                                                                                     'OOS_Date_',str(forward_date),'.pkl' ))



                                                            pickle_out_filename = os.path.join(mklOOSPredictionPathSpecific,str(symbol), oos_hash_file)
                                                            print('about to save to... ', pickle_out_filename)
                                                            pickle_out = open(pickle_out_filename, 'wb')
                                                            pickle.dump(oos_mkl_results, pickle_out)
                                                            pickle_out.close()
                                                        except (ValueError, FileNotFoundError, TypeError,PermissionError, EOFError, IndexError, FileNotFoundError, OSError, RuntimeError):
                                                            continue


                                    except (OSError, ValueError, TypeError, EOFError,FileNotFoundError, IndexError,PermissionError, FileNotFoundError, RuntimeError):

                                        continue

                    except (OSError, ValueError, TypeError, EOFError,FileNotFoundError, IndexError,PermissionError, FileNotFoundError, RuntimeError):
                        continue
            else:
                print('bad input for forward labels')
                pass



