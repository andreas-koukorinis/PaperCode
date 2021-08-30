import mkl_data_processing as mkldp
import os
import pickle
import pandas as pd
import sys
import sys
from memory_profiler import profile
import pandas as pd
import numpy as np
from collections import defaultdict
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
import time
import pickle
from collections import defaultdict
import new_alternate_single_svm as nalsvm
from mkl_data_processing import storage_location, cross_validation_results_location, evaluate_predictions, \
    oos_results_location, return_cross_val_symbol_path, model_dates_list
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier
from mkl_model_cross_validation import load_pickled_in_filename
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
# MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures
import multiprocessing
from fileutils import DataLoader as DataLoader
from fileutils import paths
from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures


# ------ functions ------- #
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


# ------- paths --------- #

mainPath = mkldp.paths('main')

symbols = sorted(os.listdir(mkldp.paths('symbols_features')))
jointFeatureLocation = os.path.join(mainPath, "ExperimentCommonLocs/JointLocationsDicts")
commonLocs = os.path.join(mainPath, "ExperimentCommonLocs/")

mklOOSPredictionPath = os.path.join(mainPath, "ExperimentCommonLocs/MKLOOSPredictions")
fittedModelsPath = os.path.join(mainPath, "ExperimentCommonLocs/FittedModels")
oosPredictionsPath = os.path.join(mainPath, "ExperimentCommonLocs/OOSPredictions")

# ---------- files

allFiles = os.listdir(jointFeatureLocation) # all files in the joint location area

labelsAlternateLocs = {f:os.path.join(commonLocs, f) for f in os.listdir(os.path.join(mainPath, "ExperimentCommonLocs/")) if str("LabelsAlternate") in f }


alternate_labels_nos = {1:"LabelsAlternateOne", 2:"LabelsAlternateTwo", 3:"LabelsAlternateThree",
                        4:"LabelsAlternateFour", 5:"LabelsAlternateFive", 6:"LabelsAlternateSix"}
# we have 6
# alternative data types


if __name__ == '__main__':
    pass
    # # pick the symbols you want to try out of sample
    # # pick the model dates- this is important as you need to calculate the out of sample "forward" dates
    # # pick label
    # alternate_label = 'LabelsAlternateFive'
    # # index of label - this can be redundant
    # alternate_label_idx = list(nalsvm.labels_pickle_files).index(alternate_label)
    # print(alternate_label)
    # # clean data location
    #
    # for symbol in ['RDSa.L']:
    #     print('Doing predictions!!')
    #     clean_data_location = storage_location(symbol)
    #     print(clean_data_location)
    #     out_of_sample_location = oos_results_location(symbol)
    #     print(symbol)
    #     # model dates list
    #     model_dates = model_dates_list(return_cross_val_symbol_path(symbol))
    #     # location of data -->dataDrive, Clean Data Storage, and label.
    #     pkl_file = load_pickled_in_filename(
    #         os.path.join(clean_data_location, os.listdir(clean_data_location)[alternate_label_idx]))
    #     date_keys = list(pkl_file.keys())
    #     # list of out of sample dates
    #
    #     for model_date in model_dates:
    #         forward_dates = nalsvm.forwardDates(date_keys, model_date)
    #
    #         print('---------------> Doing Model Date:', model_date)
    #         try:
    #             # put the features in a tensor format
    #             Xtr = normalization(rescale_01(torch.Tensor(pkl_file[model_date][0].values)))  # fitting model
    #             # put the labels in a tensor format
    #             Ytr = torch.Tensor(pkl_file[model_date][1].values)
    #             print('first bit done')
    #             # force garbage collect
    #             nalsvm.gc.collect()
    #             # kernels
    #             KLrbf = generators.RBF_generator(Xtr, gamma=[.01, .1])
    #             # dont need the next bit
    #             print('done with kernel')
    #             print(forward_dates)
    #             # base learner- use c =1 or 10
    #             # the c and lambda values need to be picked up by the cross-val results !
    #             base_learner = SVC(C=1)
    #
    #             clf = EasyMKL(lam=0.2, multiclass_strategy='ova', learner=base_learner).fit(KLrbf, Ytr)
    #             # try ovo as
    #             # well
    #             mkl_avg = AverageMKL().fit(KLrbf, Ytr)
    #             print('done')
    #             print('the combination weights are:')
    #             # this bit may be redundant here and we can put it somewhere else
    #             weights_mkl = dict()
    #             for sol in clf.solution:
    #                 print('(%d vs all): ' % sol,
    #                       clf.solution[sol].weights)  # dont need this loop- can make it redundant in another file
    #                 weights_mkl[model_date] = clf.solution[sol].weights
    #
    #             for date in forward_dates:
    #                 oos_results_mkl = defaultdict(dict)
    #                 print(date)
    #                 start = time.time()
    #                 nalsvm.logmemoryusage("Before garbage collect")
    #                 Xte = normalization(rescale_01(torch.Tensor(pkl_file[date][0].values)))
    #                 Yte = torch.Tensor(pkl_file[date][1].values)
    #                 try:
    #                     KLte = generators.RBF_generator(Xte, gamma=[.01, .1])
    #                     print('sorted out test dates bit done')
    #                     nalsvm.gc.collect()
    #                     y_pred = clf.predict(KLte)  # predictions
    #                     y_score = clf.decision_function(KLte)  # rank
    #                     # oos_svc_predictions = defaultdict(dict)
    #                     accuracy = accuracy_score(Yte, y_pred)
    #                     accuracy_file_name = "_".join((symbol, model_date, date, alternate_label, 'OOSResult.pkl'))
    #                     print('Accuracy score: %.3f' % accuracy)
    #                     evaluate_predictions(Yte, y_pred)
    #                     oos_results_mkl['MKL'][date] = evaluate_predictions(Yte, y_pred)
    #
    #                     # average kernel as a base line
    #
    #                     y_preds_average = mkl_avg.predict(KLte)  # predict the output class
    #                     y_scores_average = mkl_avg.decision_function(
    #                         KLte)  # returns the projection on the distance vector
    #                     average_accuracy = accuracy_score(Yte, y_preds_average)
    #                     print('Accuracy score: %.3f' % average_accuracy)
    #                     oos_results_mkl['AVG'][date] = evaluate_predictions(Yte, y_preds_average)
    #                 except (ValueError, TypeError, EOFError, IndexError):
    #                     continue
    #
    #         except (ValueError, TypeError, EOFError, IndexError):
    #             # at some point for clarity we need to clean these error up.
    #             continue
    #
    #         # storing the results pickle
    #
    #         pickle_out_filename = os.path.join(out_of_sample_location,
    #                                            "_".join((symbol, date, str(alternate_label), '_OOS_Results.pkl')))
    #         mkl_pickle_out = open(pickle_out_filename, 'wb')
    #         pickle.dump(oos_results_mkl, mkl_pickle_out)
    #         mkl_pickle_out.close()
    #
    #         # oos_results_mkl_df = pd.DataFrame.from_dict(oos_results_mkl)
    #         # oos_results_mkl_df.to_pickle(pickle_out_filename)
    #
    #         # storing the weights
    #
    #         weights_out_filename = os.path.join(out_of_sample_location,
    #                                             "_".join((symbol, date, str(alternate_label), '_OOS_Weights.pkl')))
    #
    #         pickle_out = open(weights_out_filename, 'wb')
    #         pickle.dump(weights_mkl, pickle_out)
    #         pickle_out.close()
    #
    #         print('Now saved: ', pickle_out_filename)
    #         nalsvm.gc.collect()
    #         print('done too')
    #         end = time.time()
    #         print(f'it took {end - start} seconds!')
