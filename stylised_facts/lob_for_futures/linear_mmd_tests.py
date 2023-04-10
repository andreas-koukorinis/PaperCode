import numpy as np
import matplotlib.pyplot as plt
import freqopttest.util as util
import freqopttest.data as data
import freqopttest.kernel as kernel
import freqopttest.tst as tst
import collections
from collections import defaultdict
import freqopttest.glo as glo
import sys
import time
import os
import pandas as pd
import pickle

mfdfaDataFrames = '/media/ak/T71/August11th2022Experiments/mfdfaDataFrames'  # Input Files
LinearMMDOutputFiles = '/media/ak/T71/August11th2022Experiments/ExperimentOne/TestMMDOutputFiles'  # Location to Save Files

symbol = 'FB1'
symbol_mfdfa_Frames_loc = os.path.join(mfdfaDataFrames, symbol)
bar_choice = 'dollar'
variable = 'mfSpect'

file_name = [f for f in os.listdir(symbol_mfdfa_Frames_loc ) if str(variable) in f][0]
file_loc = os.path.join(symbol_mfdfa_Frames_loc, file_name)
unpickled_Df = pd.read_pickle(file_loc)
len = unpickled_Df.shape[1]
print(len)
test_results_one_dict = defaultdict(dict)
test_results_two_dict = defaultdict(dict)

if __name__ == '__main__':
    if __name__ == '__main__':
        shift = 1
        window = 5  # slide
        for start_point in range(0, (len - window - shift)):
            end_point = start_point + shift

            X = np.array(unpickled_Df.iloc[:, start_point:end_point])
            Y = np.array(unpickled_Df.iloc[:, end_point + shift:end_point + 2 * shift])
            Z = np.array(unpickled_Df.iloc[:, start_point + window:end_point + window])

            data_sample = data.TSTData(X, Y)  # data to train the model
            test_data_one = data_sample = data.TSTData(X, Z)
            test_data_two = data_sample = data.TSTData(Y, Z)

            tr, te = data_sample.split_tr_te(tr_proportion=0.9, seed=100)

            # choose the best kernel that maximizes the test power
            med = util.meddistance(tr.stack_xy())

            widths = [(med * f) for f in 2.0 ** np.linspace(-1, 4, 25)]
            list_kernels = [kernel.KGauss(w ** 2) for w in widths]

            besti, powers = tst.LinearMMDTest.grid_search_kernel(tr, list_kernels, alpha=0.01)
            # print(list_kernels)
            plt.plot(widths, powers, 'o-')
            plt.xlabel('Gaussian width')
            plt.ylabel('test power')
            plt.title('median distance = %.3g. Best width: %.3g' % (med, widths[besti]))
            plt.show()
            test_data_one = data_sample = data.TSTData(X, Z)
            test_data_two = data_sample = data.TSTData(Y, Z)

            # The actual test
            best_ker = list_kernels[besti]
            lin_mmd_test = tst.LinearMMDTest(best_ker, alpha=0.01)
            # test_result_one = lin_mmd_test.perform_test(test_data_one)
            # test_result_two = lin_mmd_test.perform_test(test_data_two)

            # Test 1 Results Dict
            test_results_one_dict[start_point]['widths'] = widths
            test_results_one_dict[start_point]['med'] = med
            test_results_one_dict[start_point]['widths'] = widths
            test_results_one_dict[start_point]['besti'] = besti
            test_results_one_dict[start_point]['powers'] = powers

            test_results_one_dict[start_point]['med_on_test_data'] = util.meddistance(test_data_one.stack_xy())
            test_results_one_dict[start_point]['test_result'] = lin_mmd_test.perform_test(test_data_one)
            test_results_one_dict[start_point]['test_variance'] = lin_mmd_test.variance(X, Z, best_ker)  # test variance
            test_results_one_dict[start_point]['two_moments'] = lin_mmd_test.two_moments(X, Z,
                                                                                         best_ker)  # test variance
            test_results_one_dict[start_point]['compute_unbiased_linear_estimator'] = lin_mmd_test.compute_stat(
                test_data_one)

            # Test 2 Results Dict
            test_results_two_dict[start_point]['test_result'] = lin_mmd_test.perform_test(test_data_two)
            test_results_two_dict[start_point]['test_variance'] = lin_mmd_test.variance(Y, Z, best_ker)
            test_results_two_dict[start_point]['med_on_test_data'] = util.meddistance(
                test_data_two.stack_xy())  # test variance
            test_results_two_dict[start_point]['two_moments'] = lin_mmd_test.two_moments(Y, Z,
                                                                                         best_ker)  # test variance
            test_results_two_dict[start_point]['compute_unbiased_linear_estimator'] = lin_mmd_test.compute_stat(
                test_data_two)

            # need a better hash file below

        pickle_out_dict_one = os.path.join(LinearMMDOutputFiles, "".join(
            (str(symbol) + "_" + str(variable) + "_shift_" + str(shift) + "_wind_" + str(window) + "_" + str(
                'linear_test') + "_ONE.pkl")))
        pickle.dump(test_results_one_dict, open(pickle_out_dict_one, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        pickle_out_dict_two = os.path.join(LinearMMDOutputFiles, "".join(
            (str(symbol) + "_" + str(variable) + "_shift_" + str(shift) + "_wind_" + str(window) + "_" + str(
                'linear_test') + "_TWO.pkl")))
        pickle.dump(test_results_two_dict, open(pickle_out_dict_two, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        print(test_results_one_dict)

    # print(test_results_one_dict)
