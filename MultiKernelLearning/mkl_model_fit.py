import sys
from memory_profiler import profile

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import jsonpickle
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.metrics import pairwise
from MKLpy import generators
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.svm import SVC
from collections import OrderedDict
import os
import torch
import time
import pickle
import new_alternate_single_svm as nalsvm
from mkl_data_processing import storage_location, cross_validation_results_location, evaluate_predictions, oos_results_location, return_cross_val_symbol_path, model_dates_list
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier
from mkl_model_cross_validation import load_pickled_in_filename
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
# MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

if __name__ == '__main__':
    # pick the symbols you want to try out of sample
    # pick the model dates- this is important as you need to calculate the out of sample "forward" dates
    # pick label
    alternate_label = 'LabelsAlternateFour'
    # index of label - this can be redundant
    alternate_label_idx = list(nalsvm.labels_pickle_files).index(alternate_label)
    print(alternate_label)
    # clean data location

    for symbol in ['PRU.L']:
        clean_data_location = storage_location(symbol)
        print(symbol)
        # model dates list
        model_dates = model_dates_list(return_cross_val_symbol_path(symbol))
        # location of data -->dataDrive, Clean Data Storage, and label.
        pkl_file = load_pickled_in_filename(
            os.path.join(clean_data_location, os.listdir(clean_data_location)[alternate_label_idx]))
        date_keys = list(pkl_file.keys())
        # list of out of sample dates

        for model_date in model_dates:
            forward_dates = nalsvm.forwardDates(date_keys, model_date)

            print('---------------> Doing Model Date:', model_date)
            try:
                # put the features in a tensor format
                Xtr = normalization(rescale_01(torch.Tensor(pkl_file[model_date][0].values)))  # fitting model
                # put the labels in a tensor format
                Ytr = torch.Tensor(pkl_file[model_date][1].values)
                print('first bit done')
                # force garbage collect
                nalsvm.gc.collect()
                # kernels
                KLrbf = generators.RBF_generator(Xtr, gamma=[.01, .1])
                # dont need the next bit
                print('done with kernel')
                print(forward_dates)
                # base learner- use c =1 or 10
                # the c and lambda values need to be picked up by the cross-val results !
                base_learner = SVC(C=10)

                clf = EasyMKL(lam=0.2, multiclass_strategy='ova', learner=base_learner).fit(KLrbf, Ytr)
                # try ovo as
                # well
                mkl_avg = AverageMKL().fit(KLrbf, Ytr)
                print('done')
                print('the combination weights are:')
                # this bit may be redundant here and we can put it somewhere else
                weights_mkl = dict()
                for sol in clf.solution:
                    print('(%d vs all): ' % sol, clf.solution[sol].weights) #dont need this loop- can make it redundant in another file
                    weights_mkl[model_date] =clf.solution[sol].weights

                for date in forward_dates:
                    oos_results_mkl = dict()

                    oos_results_average = dict()
                    print(date)
                    start = time.time()
                    nalsvm.logmemoryusage("Before garbage collect")
                    Xte = normalization(rescale_01(torch.Tensor(pkl_file[date][0].values)))
                    Yte = torch.Tensor(pkl_file[date][1].values)
                    try:
                        KLte = generators.RBF_generator(Xte, gamma=[.01, .1])
                        print('sorted out test dates bit done')
                        nalsvm.gc.collect()
                        y_pred = clf.predict(KLte)  # predictions
                        y_score = clf.decision_function(KLte)  # rank
                        # oos_svc_predictions = defaultdict(dict)
                        accuracy = accuracy_score(Yte, y_pred)
                        accuracy_file_name = "_".join((symbol, model_date, date,alternate_label,'OOSResult.pkl'))
                        print('Accuracy score: %.3f' % accuracy)
                        evaluate_predictions(Yte, y_pred)
                        oos_results_mkl[date] = evaluate_predictions(Yte, y_pred)



                        # average kernel as a base line

                        y_preds_average = mkl_avg.predict(KLte)  # predict the output class
                        y_scores_average = mkl_avg.decision_function(KLte)  # returns the projection on the distance vector
                        average_accuracy = accuracy_score(Yte, y_preds_average)
                        print ('Accuracy score: %.3f' % average_accuracy)
                        evaluate_predictions(Yte, y_preds_average)
                    except (ValueError, TypeError, EOFError, IndexError):
                        continue

            except (ValueError, TypeError, EOFError, IndexError):
                # at some point for clarity we need to clean these error up.
                continue
                nalsvm.gc.collect()
                print('done too')
                end = time.time()
                print(f'it took {end - start} seconds!')
