import numpy as np
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
import itertools
from collections import defaultdict
import pandas as pd
import freqopttest.util as util
import freqopttest.data as data
import freqopttest.kernel as kernel
import freqopttest.tst as tst

import freqopttest.glo as glo
import sys
import time
import itertools

def generate_column_pairs(num_columns):
    """
    Generates pairs of columns for sliding window analysis.

    Args:
        num_columns (int): The total number of columns.

    Yields:
        tuple: A tuple containing the start index, end index, and shift for a pair of columns.
               The start and end indices define the column pair, and the shift indicates the distance
               by which one column is shifted with respect to the other.
    """
    for window in range(1, num_columns):
        for shift in range(1, num_columns - window + 1):
            for start_point, end_point in itertools.combinations(range(num_columns), 2):
                if end_point - start_point == window:
                    yield start_point, end_point, shift


def create_kgauss(sigma2, default_sigma2=None):
    """
    Create a KGauss instance with the given sigma2 value, or use the default_sigma2 value if provided.

    :param sigma2: float, the sigma2 value to use for creating the KGauss instance.
    :param default_sigma2: float, optional, the default sigma2 value to use if the provided sigma2 is invalid.
    :return: KGauss, the created KGauss instance.
    :raise ValueError: if both sigma2 and default_sigma2 are invalid.
    """
    if sigma2 > 0:
        return kernel.KGauss(sigma2)
    elif default_sigma2 is not None and default_sigma2 > 0:
        print("Using default sigma2 value:", default_sigma2)
        return kernel.KGauss(default_sigma2)
    else:
        raise ValueError("Both sigma2 and default_sigma2 are invalid. Please provide a positive value for either.")


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
    try:
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
    except np.linalg.LinAlgError as e:
        print("LinAlgError:", str(e))
        return np.array([])  # Return an empty array or any other appropriate value

if __name__ == "__main__":
    reconLOBs = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB'
    figuresLocation = '/home/ak/Documents/Research/Papers/StylisedFactsPaper/figures'
    symbols = sorted(os.listdir(reconLOBs))
    symbolFolders = [os.path.join(reconLOBs, symbol) for symbol in symbols]
    # os.listdir('/media/ak/Data1/InterestRateFuturesData/')
    directory = 'OrderBookImbalance'
    path = os.path.join(reconLOBs, directory, 'extremes', )
    # Assuming you have a list of file paths
    obiFiles = os.listdir(path)
    file_paths = [os.path.join(path, obiFiles[i]) for i in range(7)]
    obimfdfaPath = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance/OBIMFDFA'
    # Keys of interest
    mfdfakeys = ['orderBookImbalance', 'n', 'F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect', 'qs']

    # Initialize a nested defaultdict
    data_dict = defaultdict(lambda: defaultdict(dict))
    files = [f for f in os.listdir(obimfdfaPath) if str('_obi_mfdfa') in f]
    # Iterate over files and aggregate data
    for filename in files:
        symbol = filename.split("_")[0]  # Extracting symbol from filename
        #     print(symbol)
        file_path = os.path.join(obimfdfaPath, filename)
        unpickledData = pd.read_pickle(file_path)  # Assuming data is a dict of dicts
        keys = list(unpickledData.keys())
            # Iterate over each key and index
        for idx in keys:
            for key in mfdfakeys:
                data_dict[symbol][key][idx] = unpickledData[idx][key]
    # Assuming `a_args` and `second_arg` are defined
    mmd_train_test_results = defaultdict(dict)
    a_args = [1, 2, 3]  # Example values
    second_arg = [1, 4, 5, 6, 8, 10]  # Example values
    #
    c = list(itertools.product(a_args, second_arg))
    mfdfakey = 'list_H'
    for symbol in ['DU1', 'FB1', 'JB1', 'OE1', 'FV1', 'G', 'KE1', 'RX1', 'US1', 'XM1', 'YM1']:
        print('Doing Symbol: %s' % symbol)
        choice_dataframe = pd.DataFrame.from_dict(data_dict[str(symbol)][str(mfdfakey)])
        # Replace 'len' with a different variable name, like 'length'
        length = choice_dataframe.shape[1]  # This should be set to your actual length value

        for shift, window in c:
            for start_point in range(0, (length - window - shift)):
                end_point = start_point + shift
                X = np.array(choice_dataframe.iloc[:, start_point:end_point])
                Y = np.array(choice_dataframe.iloc[:, end_point + shift:end_point + 2 * shift])
                Z = np.array(choice_dataframe.iloc[:, start_point + window:end_point + window])
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

                pickle_out_dict_one = os.path.join('/media/ak/Data/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance/OBIMFDFAResults',
                                                   "".join(
                    (str(symbol) + "_" + str(mfdfakey) + "_shift_" + str(shift) + "_wind_" + str(window) + "_" + str(
                        'OBI_quad_MMD_test') + ".pkl")))
                pickle.dump(mmd_train_test_results, open(pickle_out_dict_one, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



