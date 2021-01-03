import sys
from memory_profiler import profile

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
sys.path.append('/home/ak/Documents/Research/PaperCode/MultiKernelLearning')
import jsonpickle
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.metrics import pairwise
import MKLpy
from MKLpy import generators
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.svm import SVC
from collections import OrderedDict
import os
import torch
import time
import pickle
import new_alternate_single_svm as nalsvm
from mkl_data_processing import storage_location, cross_validation_results_location, evaluate_predictions, \
    oos_results_location, return_cross_val_symbol_path, model_dates_list, pickle_dump_obj_to_filename
from MKLpy.multiclass import OneVsRestMKLClassifier, OneVsOneMKLClassifier
from MKLpy.metrics import margin, radius, ratio, trace, frobenius
from MKLpy.preprocessing import kernel_normalization
from mkl_model_cross_validation import load_pickled_in_filename
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
# MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM


def kernel_evaluation(K):
    kernel_results_dict = {}
    K = kernel_normalization(K)  # normalize the kernel K (useless in the case of HPK computed on normalized data)
    kernel_results_dict['score_margin'] = margin(K,
                                                 Ytr)  # the distance between the positive and negative classes in the kernel space
    kernel_results_dict['score_radius'] = radius(
        K)  # the radius of the Einimum Enclosing Ball containing data in the kernel space
    kernel_results_dict['score_ratio'] = ratio(K,
                                               Ytr)  # the radius/margin ratio defined as (radius**2/margin**2)/n_examples
    kernel_results_dict['score_froben'] = frobenius(K)  # the Frobenius norm of a kernel matrix
    kernel_results_dict['score_trace'] = trace(K)  # the trace of the kernel matrix
    return kernel_results_dict


# the margin between positive and negative classes in the kernel space;
# the radius of the Minimum Enclosing Ball (MEB) containing data in the kernel space;
# the radius/margin ratio;
# the trace of a kernel matrix;
# the frobenius norm of a kernel matrix.

if __name__ == '__main__':
    # pick the symbols you want to try out of sample
    # pick the model dates- this is important as you need to calculate the out of sample "forward" dates
    # pick label
    alternate_label = 'LabelsAlternateOne'
    # index of label - this can be redundant
    alternate_label_idx = list(nalsvm.labels_pickle_files).index(alternate_label)
    print(alternate_label)
    # clean data location

    for symbol in ['SHM.L']:
        clean_data_location = storage_location(symbol)
        print(symbol)
        # model dates list
        model_dates = model_dates_list(return_cross_val_symbol_path(symbol))
        # location of data -->dataDrive, Clean Data Storage, and label.
        pkl_file = load_pickled_in_filename(
            os.path.join(clean_data_location, os.listdir(clean_data_location)[alternate_label_idx]))
        date_keys = list(pkl_file.keys())
        # model_date = '20170704'
        # list of out of sample dates
        for model_date in model_dates:

            start = time.time()
            # forward_dates = nalsvm.forwardDates(date_keys, model_date)

            print('---------------> Doing Model Date:', model_date)

            # put the features in a tensor format
            Xtr = rescale_01(torch.Tensor(pkl_file[model_date][0].values))
            Xtr = normalization(Xtr)  # fitting model
            # put the labels in a tensor format
            Ytr = torch.Tensor(pkl_file[model_date][1].values)
            #
            try:

                KLtr_poly = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(6)]
                deg = 5
                K = KLtr_poly[deg]  # the HPK with degree 5
                # K is always a squared kernel matrix, i.e. it is not the kernel computed between test and training
                # examples.
                kernel_evaluation_dict = kernel_evaluation(K)
                print('done')
                print('results of the %d-degree HP kernel:' % deg)
                print('margin: %.4f, radius: %.4f, radiu-margin ratio: %.4f,' % (kernel_evaluation_dict['score_margin'], kernel_evaluation_dict['score_radius'], kernel_evaluation_dict['score_ratio']))
                print('trace: %.4f, frobenius norm: %.4f' % (kernel_evaluation_dict['score_trace'], kernel_evaluation_dict['score_froben']))
                kernel_evaluation_results  = dict()
                kernel_evaluation_results[model_date] = kernel_evaluation_dict

            except(MKLpy.utils.exceptions.BinaryProblemError):
                    pass

        destinationPath = cross_validation_results_location(symbol)
        kernel_type= 'homogeneous_polynomial_kernel'
        fileName = "_".join((alternate_label, kernel_type+'.pkl'))
        pickle_dump_obj_to_filename(destinationPath, symbol, fileName, kernel_evaluation_results)

