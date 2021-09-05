import mkl_data_processing as mkldp
import os
import pickle
import pandas as pd
import sys
import sys
from memory_profiler import profile
import pandas as pd
import numpy as np

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import jsonpickle
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.metrics import pairwise
from MKLpy import generators
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.svm import SVC
from collections import OrderedDict
import os
from sklearn.metrics import accuracy_score
import torch
from MKLpy.model_selection import cross_val_score
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier
from sklearn.model_selection import train_test_split
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD, HeuristicMKL, PWMK, FHeuristic, CKA  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM
from MKLpy.metrics import pairwise
from MKLpy.utils.misc import identity_kernel
from fileutils import DataLoader as DataLoader
from fileutils import paths
from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures
from MKLpy.model_selection import cross_val_score
from sklearn.svm import SVC
import multiprocessing

from collections import defaultdict
import os
import multiprocessing
import psutil
import pickle
from collections import defaultdict
import itertools
import time

from MKLpy.scheduler import ReduceOnWorsening
from MKLpy.callbacks import EarlyStopping
import matplotlib.pyplot as plt


### functions ###

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


def logmemoryusage(msg):
    # function to log memory usage
    process = psutil.Process(os.getpid())
    print('Memory usage at %s is %smb.' % (msg, process.memory_info().rss / 1000 / 1000))


### paths ###

mainPath = mkldp.paths('main')

symbols = sorted(os.listdir(mkldp.paths('symbols_features')))
jointFeatureLocation = os.path.join(mainPath, "ExperimentCommonLocs/JointLocationsDicts")
commonLocs = os.path.join(mainPath, "ExperimentCommonLocs/")
# common locs contains a lot of folders, including the prediction targets and the rest

mklOOSPredictionPath = os.path.join(mainPath, "ExperimentCommonLocs/MKLOOSPredictions")
allFiles = os.listdir(jointFeatureLocation)
fittedModelsPath = os.path.join(mainPath, "ExperimentCommonLocs/FittedModels")
oosPredictionsPath = os.path.join(mainPath, "ExperimentCommonLocs/OOSPredictions")

labelsAlternateLocs = {f: os.path.join(commonLocs, f) for f in
                       os.listdir(os.path.join(mainPath, "ExperimentCommonLocs/")) if str("LabelsAlternate") in f}
labelKeys = list(labelsAlternateLocs.keys())
alternate_labels_nos = {1: "LabelsAlternateOne", 2: "LabelsAlternateTwo", 3: "LabelsAlternateThree",
                        4: "LabelsAlternateFour", 5: "LabelsAlternateFive",
                        6: "LabelsAlternateSix"}  # we have 7 alternative data types


def outer_loop_function(hmm_date):
    # this is the usual loop but we are writing it as a function so I can parallelise it

    hmm_features_date_path = os.path.join(symbolData.symbol_features_path, hmm_date)
    symbol_feature_paths = {feature_file.split("_")[5]: os.path.join(hmm_features_date_path, feature_file) for
                            f, feature_file in enumerate(sorted(os.listdir(hmm_features_date_path)))}
    keys = sorted(list(symbol_feature_paths.keys()))
    print('For hmm date: ', hmm_date, '###########')
    for key in keys:

        print('For key: ', key, '############')

        labels_file_path = os.path.join(symbolData.symbol_specific_label_path(label_idx), key + ".csv")
        output_dict = defaultdict(dict)

        if os.path.isfile(labels_file_path):  # check that this is a real path
            print("can train")
            print(" reading labels")  # this is the labels path!
            labels = pd.read_csv(labels_file_path)
            label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
            logmemoryusage("Before garbage collect")
            hmm_features = nfu.hmm_features_df(open_pickle_filepath(symbol_feature_paths[key]))
            if hmm_features.isnull().values.all():  # checking that the HMM features are actually not null
                continue
                print('lots of NaNs on features')
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

                y_train = df.reindex(columns =df.columns[df.columns.str.contains(pat='label')])   # training labels
                if df_final.shape[0] < 10:
                    print(' the ratio of classes is too low. try another label permutation')
                    problem_dict[hmm_date][key] = str(key)
                    continue
                else:
                    try:
                        print("starting model fit")
                        Xtr, Xte, Ytr, Yte = train_test_split(df_final, y_train, test_size=.5, random_state=42)
                        # training
                        arrXtr = np.array(Xtr)
                        X_tr = normalization(rescale_01(arrXtr))
                        Y_tr = torch.Tensor(Ytr.values.ravel())

                        # testing

                        arrXte = np.array(Xte)
                        X_te = normalization(rescale_01(arrXte))
                        Y_te = torch.Tensor(Yte.values.ravel())

                        KLtr = [pairwise.homogeneous_polynomial_kernel(X_tr, degree=d) for d in range(1, 21)] + [
                            identity_kernel(len(Ytr))]
                        KLte = [pairwise.homogeneous_polynomial_kernel(X_te, X_tr, degree=d) for d in range(1, 21)]
                        KLte.append(torch.zeros(KLte[0].size()))

                        print('tuning lambda for EasyMKL...', end='')
                        base_learner = SVC(C=0.1)  # "soft"-margin svm
                        best_results = {}
                        lam = 0.1  # possible lambda values for the EasyMKL algorithm
                        # MKLpy.model_selection.cross_val_score performs the cross validation automatically,
                        # it may returns accuracy, auc, or F1 scores
                        scores = cross_val_score(KLtr, Y_tr, EasyMKL(learner=base_learner, lam=lam), n_folds=5,
                                                 scoring='accuracy')
                        acc = np.mean(scores)
                        # if not best_results or best_results['score'] < acc:
                        #     best_results = {'lam': lam, 'score': acc}
                        # evaluation on the test set
                        from sklearn.metrics import accuracy_score
                        print('done')
                        clf = EasyMKL(learner=base_learner, lam=lam).fit(KLtr, Ytr)
                        y_pred = clf.predict(KLte)
                        accuracy = accuracy_score(Y_te, y_pred)
                        print('accuracy on the test set: %.3f, with lambda=%.2f' % (accuracy, best_results['lam']))
                    except ValueError:
                        continue
                        logmemoryusage("at the end")

        else:
            print("#################### Your Labels File does not exist ----- ####")
        continue
        logmemoryusage("Before garbage collect")


if __name__ == '__main__':
    symbol_idx = 16  # pick a symbol - will need to iterate over a number of symbols
    symbol = sorted(symbols)[symbol_idx]  # to be serialised so read all the symbols
    print(symbol)

    symbolData = DataLoader(mainPath, symbol)  # initiate a path where all the data should be

    label_idx = 1  # takes number 1 to 3

    labelsLocation = labelsAlternateLocs[alternate_labels_nos[label_idx]]

    labelsPath = os.path.join(labelsLocation, symbol)

    start_time = time.time()

    with multiprocessing.Pool(processes=8) as process_pool:
        process_pool.map(outer_loop_function, sorted(symbolData.hmm_dates_list))
    current_time = time.time()
    elapsed_time = current_time - start_time
    print('elapsed time', elapsed_time)
