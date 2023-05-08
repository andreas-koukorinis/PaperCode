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
import itertools
mfdfaDataFrames = '/media/ak/T7/August11th2022Experiments/mfdfaDataFrames'  # Input Files
LinearMMDOutputFiles = '/media/ak/T7/August11th2022Experiments/ExperimentOne/TestMMDOutputFiles'  # Location to Save Files


def simulate_null_spectral(weights, n_simulate=1000, seed=275):
    """
    weights: chi-square weights (for the infinite weigted sum of chi squares)
    Return the values of MMD^2 (NOT n*MMD^2) simulated from the null distribution by
    the spectral method.
    """
    # draw at most block_size values at a time
    block_size = 400
    D = weights.shape[0]  # len(weights)
    mmds = np.zeros(n_simulate)
    from_ind = 0

    with util.NumpySeedContext(seed=seed):
        while from_ind < n_simulate:
            to_draw = min(block_size, n_simulate - from_ind)
            # draw chi^2 random variables.
            chi2 = np.random.randn(D, to_draw) ** 2
            # an array of length to_draw
            sim_mmds = 2.0 * weights.dot(chi2 - 1.0)
            # store
            end_ind = from_ind + to_draw
            mmds[from_ind:end_ind] = sim_mmds
            from_ind = end_ind
    return mmds


def chi_square_weights_H0(k, X):
    """
    Return a numpy array of the weights to be used as the weights in the
    weighted sum of chi-squares for the null distribution of MMD^2.
    - k: a Kernel
    - X: n x d number array of n data points
    """
    n = X.shape[0]
    # Gram matrix
    K = k.eval(X, X)
    # centring matrix. Not the most efficient way.
    H = np.eye(n) - np.ones((n, n)) / float(n)
    HKH = H.dot(K).dot(H)
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.eigvals.html
    evals = np.linalg.eigvals(HKH)
    evals = np.real(evals)
    # sort in decreasing order
    evals = -np.sort(-evals)
    weights = evals / float(n) ** 2
    return weights
#-------# #----------------# FILES & Locations #-----------------#
symbol = 'XM1'
symbol_mfdfa_Frames_loc = os.path.join(mfdfaDataFrames, symbol)
bar_choice = 'dollar'
variable = 'alpha'


t7 ='/media/ak/T7'
experimentFiles = os.path.join(t7,'August11th2022Experiments')

file_name = [f for f in os.listdir(symbol_mfdfa_Frames_loc ) if str(variable) in f][0]
file_loc = os.path.join(symbol_mfdfa_Frames_loc, file_name)
unpickled_Df = pd.read_pickle(file_loc)
len = unpickled_Df.shape[1]
print(len)

mmd_train_test_results = defaultdict(dict)

if __name__ == '__main__':

    start = time.time()
    a_args =[1, 2, 3]
    second_arg = [1, 4, 5, 6, 8, 10]
    c = list(itertools.product(a_args, second_arg))
    for shift, window in c:
        for start_point in range(0, (len - window - shift)):
            end_point = start_point + shift

            X = np.array(unpickled_Df.iloc[:, start_point:end_point])
            Y = np.array(unpickled_Df.iloc[:, end_point + shift:end_point + 2 * shift])
            Z = np.array(unpickled_Df.iloc[:, start_point + window:end_point + window])

            try:
                tr_data = data.TSTData(X, Y)
                test_data_one = data.TSTData(X, Z)  # data to train the model
                test_data_two = data.TSTData(Y, Z)
                # training dictionary results
                tr, te = tr_data.split_tr_te(tr_proportion=0.95, seed=10)  # is this necessary?!

                xtr, ytr = tr.xy()
                xytr = tr.stack_xy()
                sig2 = util.meddistance(xytr, subsample=1000)
                k = kernel.KGauss(sig2)
                mean, var = tst.QuadMMDTest.h1_mean_var(xtr, ytr, k, is_var_computed=True)
                Kx = k.eval(xtr, xtr)
                Ky = k.eval(ytr, ytr)
                Kxy = k.eval(xtr, ytr)
                mean_gram, var_gram = tst.QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, k, True)
                chi2_weights = chi_square_weights_H0(k, xytr)
                sim_mmds = simulate_null_spectral(chi2_weights, n_simulate=2000)
                # choose the best parameter and perform a test with permutations
                med = util.meddistance(tr.stack_xy(), 1000)
                list_gwidth = np.hstack(((med ** 2) * (2.0 ** np.linspace(-4, 4, 20))))
                list_gwidth.sort()
                list_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

                # grid search to choose the best Gaussian width
                besti, powers = tst.QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha=0.05)
                # perform test
                best_ker = list_kernels[besti]
                mmd_train_test_results[start_point]['perm_mmds1'] = tst.QuadMMDTest.permutation_list_mmd2(xtr, ytr, k,
                                                                                                          n_permute=2000)
                mmd_train_test_results[start_point]['chi2_weights'] = chi2_weights
                mmd_train_test_results[start_point]['sim_mmds'] = sim_mmds
                mmd_train_test_results[start_point]['sig2'] = sig2
                mmd_train_test_results[start_point]['Kxy'] = k.eval(xtr, ytr)
                mmd_train_test_results[start_point]['mean'] = mean
                mmd_train_test_results[start_point]['var'] = var
                mmd_train_test_results[start_point]['Kxx'] = k.eval(xtr, xtr)
                mmd_train_test_results[start_point]['Kyy'] = k.eval(ytr, ytr)
                mmd_train_test_results[start_point]['mean_gram'] = mean_gram
                mmd_train_test_results[start_point]['var_gram'] = var_gram
                mmd_train_test_results[start_point]['med'] = util.meddistance(tr.stack_xy(), 1000)
                mmd_train_test_results[start_point]['list_gwidth'] = list_gwidth.sort()
                mmd_train_test_results[start_point]['list_kernels'] = list_kernels
                mmd_train_test_results[start_point]['besti'] = besti
                mmd_train_test_results[start_point]['powers'] = powers
                mmd_train_test_results[start_point]['best_ker'] = best_ker

                alpha = 0.05
                mmd_test = tst.QuadMMDTest(best_ker, n_permute=2000, alpha=alpha)
                mmd_train_test_results[start_point]['XZ_test'] = mmd_test.perform_test(test_data_one)
                mmd_train_test_results[start_point]['YZ_test'] = mmd_test.perform_test(test_data_two)
            except ValueError:
                pass

                # testing the first data set
            pickle_out_dict_one = os.path.join(LinearMMDOutputFiles, "".join(
                (str(symbol) + "_" + str(variable) + "_shift_" + str(shift) + "_wind_" + str(window) + "_" + str(
                    'quad_MMD_test') + ".pkl")))
            pickle.dump(mmd_train_test_results, open(pickle_out_dict_one, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print(end - start)
