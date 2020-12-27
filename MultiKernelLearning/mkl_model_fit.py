import sys
from memory_profiler import profile

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import jsonpickle
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.metrics import pairwise
# from MKLpy.metrics import margin, radius, ratio, trace, frobenius
# from MKLpy.preprocessing import kernel_normalization
from MKLpy import generators
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.svm import SVC
from MKLpy.model_selection import cross_val_score
from MKLpy.generators import HPK_generator, RBF_generator
import pandas as pd
from collections import defaultdict
from itertools import product
import os
import torch
import time
import pickle
import new_alternate_single_svm as nalsvm
from mkl_data_processing import storage_location, cross_validation_results_location
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier
from mkl_model_cross_validation import load_pickled_in_filename
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
# MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

if __name__ == '__main__':
    for symbol in ['ECM.L', 'SHP.L']:
        model_date = '20170705'
        alternate_label = 'LabelsAlternateFour'
        print(alternate_label)
        clean_data_location = storage_location(symbol)
        alternate_label_idx = list(nalsvm.labels_pickle_files).index(alternate_label)
        pkl_file = load_pickled_in_filename(
            os.path.join(clean_data_location, os.listdir(clean_data_location)[alternate_label_idx]))
        date_keys = list(pkl_file.keys())
        forward_dates = nalsvm.forwardDates(date_keys, model_date)

        print('--------------->')
        try:
            Xtr = normalization(rescale_01(torch.Tensor(pkl_file[model_date][0].values)))  # fitting model
            Ytr = torch.Tensor(pkl_file[model_date][1].values)
            print('first bit done')
            nalsvm.gc.collect()
            KLrbf = generators.RBF_generator(Xtr, gamma=[.001, .01, .1])
            print('done with kernel')
            print(forward_dates)
            base_learner = SVC(C=1)
            clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLrbf, Ytr)
            mkl_avg = AverageMKL().fit(KLrbf, Ytr)
            print('done')
            print('the combination weights are:')
            for sol in clf.solution:
                print('(%d vs all): ' % sol, clf.solution[sol].weights)

            for date in forward_dates:
                print(date)
                start = time.time()
                nalsvm.logmemoryusage("Before garbage collect")
                Xte = normalization(rescale_01(torch.Tensor(pkl_file[date][0].values)))
                Yte = torch.Tensor(pkl_file[date][1].values)
                try:
                    KLte = generators.RBF_generator(Xte, gamma=[.001, .01, .1])
                    print('sorted out test dates bit done')
                    nalsvm.gc.collect()
                    y_pred = clf.predict(KLte)  # predictions
                    y_score = clf.decision_function(KLte)  # rank
                    accuracy = accuracy_score(Yte, y_pred)
                    print('Accuracy score: %.3f' % accuracy)

                    y_preds_average = mkl_avg.predict(KLte)  # predict the output class
                    y_scores_average = mkl_avg.decision_function(KLte)  # returns the projection on the distance vector
                    average_accuracy = accuracy_score(Yte, y_preds_average)
                    print ('Accuracy score: %.3f' % average_accuracy)
                except (ValueError, TypeError, EOFError, IndexError):
                    continue

        except (ValueError, TypeError, EOFError, IndexError):
            continue

        #     try:
        #         lam_values = [0.1]
        #         C_values = [1]
        #         print(C_values)
        #         for lam, C in product(lam_values, C_values):
        #             print('now here', C, lam)
        #             svm = SVC(C=C)
        #             mkl = EasyMKL(lam=lam, learner=svm)
        #             scores = cross_val_score(KLrbf, Ytr, mkl, n_folds=3, scoring='accuracy')
        #             print(str(scores))
        #             print(lam, C, scores)
        #             print(type(scores))
        #
        #             nalsvm.logmemoryusage("Before garbage collect")
        #             print('---------------> moving on')
        #
        #     except (ValueError, TypeError, EOFError):
        #         continue
        #     # only way that seems to work for this
        #     # pickle_out_filename = os.path.join(cross_validation_data_location, "_".join((symbol,date,'RBF_CrossValidationResults.pkl')))
        #     # test_df =pd.DataFrame.from_dict(cv_dict_list)
        #     # test_df.to_pickle(pickle_out_filename)
        #     #
            nalsvm.gc.collect()
            print('done too')
            end = time.time()
            print(f'it took {end - start} seconds!')
