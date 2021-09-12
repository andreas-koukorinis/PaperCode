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

"""
piece of code that does cross validation and stores the results
"""


def load_pickled_in_filename(file):
    # load a simple pickled file and return it. its a bit different to the method used for the dictionary as this
    # is pure Python 3.x
    pickle_in = open(file, 'rb')
    return pickle.load(pickle_in)


if __name__ == '__main__':
    cv_dict_list = defaultdict(dict)
    for symbol in ['ECM.L']:
        print(symbol)  # which symbol - unnecessary at this point
        cross_validation_data_location = cross_validation_results_location(symbol)
        clean_data_location = storage_location(symbol)
        for alternate_label_idx in range(0, 4):
            alternate_label = nalsvm.labels_pickle_files[alternate_label_idx]
            print(alternate_label)
            file_to_load = os.path.join(clean_data_location, os.listdir(clean_data_location)[alternate_label_idx])
            pkl_file = load_pickled_in_filename(file_to_load)
            date_keys = list(pkl_file.keys())
            print('--------------->')
            for date in date_keys:  # date is model fit-date i.e the date we pick up to fit the training model in CV
                print(date)
                start = time.time()
                nalsvm.logmemoryusage("Before garbage collect")
                Xtr = normalization(rescale_01(torch.Tensor(pkl_file[date][0].values)))
                Ytr = torch.Tensor(pkl_file[date][1].values)
                print('first bit done')
            nalsvm.gc.collect()
                KLrbf = generators.RBF_generator(Xtr, gamma=[.001, .01, .1])
                print('done with kernel')
                nalsvm.gc.collect()
                try:
                    lam_values = [0, 0.1, 0.2, 1]
                    C_values = [0.01, 1, 10, 100]
                    print(C_values)
                    for lam, C in product(lam_values, C_values):
                        print('now here', C, lam)
                        svm = SVC(C=C)
                        mkl = EasyMKL(lam=lam, learner=svm)
                        scores = cross_val_score(KLrbf, Ytr, mkl, n_folds=3, scoring='accuracy')
                        print(str(scores))
                        print(lam, C, scores)
                        print(type(scores))
                        cv_dict_list[(symbol, date, alternate_label)][(lam, C)] = scores
                        nalsvm.logmemoryusage("Before garbage collect")
                        print('---------------> moving on')

                except (ValueError, TypeError, EOFError):
                    continue
                # only way that seems to work for this
                pickle_out_filename = os.path.join(cross_validation_data_location,
                                                   "_".join((symbol, date, 'RBF_CrossValidationResults.pkl')))
                test_df = pd.DataFrame.from_dict(cv_dict_list)
                test_df.to_pickle(pickle_out_filename)
                print('Now saved: ', pickle_out_filename)

                nalsvm.gc.collect()
                print('done too')
                end = time.time()
                print(f'it took {end - start} seconds!')
