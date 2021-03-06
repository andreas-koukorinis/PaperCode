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

from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

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
    allFiles = os.listdir(jointFeatureLocation)
    # needs to be parallelised from here

    for file in allFiles:  # pick a file- any file

        select_file_path = os.path.join(jointFeatureLocation, file)  # formulate the path
        print('Symbol:----->', file.split("_")[0])

        select_hmm_date = select_file_path.split("_")[3]  # pull out the hmm_date - strip it out

        select_feature_label_date = select_file_path.split("_")[6]  # pull out the label_feature_date

        select_label_idx = select_file_path.split("_")[9]  # pull out the label _idx

        unpickled_select_file = open_pickle_filepath(select_file_path)  # unplickle the select file

        hmm_keys = sorted(list(unpickled_select_file.keys()))  # hmm keys for the select file.

        for hmm_date_key in hmm_keys:  # pick and hmm date
            feature_label_keys = sorted(
                unpickled_select_file[hmm_date_key].keys())  # each key here unlocks a feature and label set

            for feature_label_date in feature_label_keys:  # make a list of all the feature dates
                features_file_path = unpickled_select_file[hmm_date_key][feature_label_date][
                    0]  # this is the feature path
                labels_file_path = unpickled_select_file[hmm_date_key][feature_label_date][1]  # this is the labels path

                if os.path.isfile(features_file_path):  # if label file exists I can traing
                    print('ok----->', feature_label_date)  # if you got to this point we have data so we can mov eon
                    labels = pd.read_csv(labels_file_path)  # open labels file
                    label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                    features = open_pickle_filepath(features_file_path)  # opens features file
                    hmm_features = nfu.hmm_features_df(features)  # get the hmm features out, so unpack the tuples!
                    print('loaded features and labels ')
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
                        if df_final.shape[0] < 10:  # make sure it all looks reasonable
                            print(' the ratio of classes is too low. try another label permutation')
                            continue
                        else:

                            print("starting model fit")
                            # put the features in a tensor format
                            X = np.asarray(df_final.values)  # need this for torch
                            Xtr = normalization(rescale_01(torch.Tensor(X)))  # features in a tensor format

                            Ytr = torch.Tensor(y_train.values)  # put the labels in a tensor format
                            print('-----------------first bit done------------------')
                            KLrbf = generators.RBF_generator(Xtr, gamma=[.01,
                                                                         .1])  # get a few RBF Kernels ready - maybe need more here
                            print('done with kernel')

                            #             # base learner- use c =1 or 10
                            #             # the c and lambda values need to be picked up by the cross-val results !



                            C_range = [0.1, 1, 10, 100]
                            lam_range = [0, 0.01, 0.1, 0.2, 0.9]
                            # parallelization should start here
                            # with multiprocessing.Pool(processes=4) as process_pool:
                            #     process_pool.starmap(mkl_learning_parallel, itertools.product(C_range, lam_range))

                            def mkl_learning_parallel(C_choice, lam):
                                best_results = {}
                            # for C_choice in C_range:
                                base_learner = SVC(C=C_choice)  # "hard"-margin svm
                                # clf = EasyMKL(lam=0.2, multiclass_strategy='ova', learner=base_learner).fit(KLrbf,
                                #                                                                             Ytr)
                                # print('done')
                                # print('the combination weights are:')
                                #
                                # for sol in clf.solution:
                                #     print('(%d vs all): ' % sol,
                                #           clf.solution[
                                #               sol].weights)  # need to store these results somewhere

                                # for lam in lam_range:  # possible lambda values for the EasyMKL algorithm
                                    # MKLpy.model_selection.cross_val_score performs the cross validation
                                    # automatically, it may returns accuracy, auc, or F1 scores
                                    scores = cross_val_score(KLrbf, Ytr, EasyMKL(learner=base_learner, lam=lam),
                                                             n_folds=5,
                                                             scoring='accuracy')  # get the cross-validation scores
                                    acc = np.mean(scores)
                                    if not best_results or best_results['score'] < acc:
                                        best_results = {'C': C_choice, 'lam': lam,
                                                        'score': acc}  # these should get dumped somewhere
                                    print('done')
                                    # anything below this point is kind of redundant -its a simple evaluation of the solution!
                                    best_learner = SVC(C=best_results['C'])
                                    clf = EasyMKL(learner=best_learner, lam=best_results['lam']).fit(KLrbf, Ytr)
                                    y_pred = clf.predict(KLrbf)
                                    accuracy = accuracy_score(Ytr, y_pred)
                                    print('accuracy on the test set: %.3f, with lambda=%.2f' % (
                                        accuracy, best_results['lam']))


                            # except ValueError:
                            #     continue


                else:
                    print('PROBLEM----->in one of of your locations')

                # forward_Dates_List =(sorted(forward_Dates(feature_label_keys, feature_label_date)))
