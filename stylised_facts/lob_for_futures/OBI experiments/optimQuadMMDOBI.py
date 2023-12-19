import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
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
import json
import os


### Functions ###

def save_data_slices(data_dict, obimfdfaPath, mfdfakeys):
    """
    Save slices of data_dict to individual files.

    Args:
        data_dict (dict): The data dictionary to be sliced and saved.
        obimfdfaPath (str): Path to save the sliced data files.
        mfdfakeys (list): List of mfdfakeys to consider.
    """
    for symbol in data_dict:
        for mfdfakey in mfdfakeys:
            # Construct the file path
            file_path = os.path.join(obimfdfaPath, f"{symbol}_{mfdfakey}_data.json")
            # Save the slice of data_dict to a file
            with open(file_path, 'w') as file:
                json.dump(data_dict[symbol][mfdfakey], file)
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
    ################################
def save_data_slices(data_dict, obimfdfaPath, mfdfakeys):
    """
    Save slices of data_dict to individual files.

    Args:
        data_dict (dict): The data dictionary to be sliced and saved.
        obimfdfaPath (str): Path to save the sliced data files.
        mfdfakeys (list): List of mfdfakeys to consider.
    """
    for symbol in data_dict:
        for mfdfakey in mfdfakeys:
            # Construct the file path
            file_path = os.path.join(obimfdfaPath, f"{symbol}_{mfdfakey}_data_slice.json")
            # Convert the data to a JSON-serializable format
            data_to_save = {idx: data.to_dict() if isinstance(data, pd.Series) else data
                            for idx, data in data_dict[symbol][mfdfakey].items()}
            # Save the slice of data_dict to a file
            with open(file_path, 'w') as file:
                json.dump(data_to_save, file)
def process_data_from_file(symbol, mfdfakey, file_path, obimfdfaResultsPath, a_args, second_arg):
    """
    Process data for a given symbol and mfdfakey, reading data from a file. The Worker Function.

    Args:
        symbol (str): The symbol to process.
        mfdfakey (str): The mfdfakey to process.
        file_path (str): Path to the file containing the data.
        obimfdfaPath (str): Path where the results will be saved.
        a_args (list): List of first arguments for analysis.
        second_arg (list): List of second arguments for analysis.
    """
    # Load the data from the file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Convert the data for the symbol and mfdfakey to a DataFrame
    choice_dataframe = pd.DataFrame.from_dict(data_dict[symbol][mfdfakey])
    length = choice_dataframe.shape[1]
    mmd_train_test_results = defaultdict(dict)

    # Iterate over combinations of shift and window
    for shift, window in itertools.product(a_args, second_arg):
        for start_point in range(0, (length - window - shift)):
            end_point = start_point + shift
            X = np.array(choice_dataframe.iloc[:, start_point:end_point])
            Y = np.array(choice_dataframe.iloc[:, end_point + shift:end_point + 2 * shift])
            Z = np.array(choice_dataframe.iloc[:, start_point + window:end_point + window])

            try:
                tr_data = data.TSTData(X, Y)
                test_data_one = data.TSTData(X, Z)  # data to train the model
                test_data_two = data.TSTData(Y, Z)

                tr, te = tr_data.split_tr_te(tr_proportion=0.95, seed=10)

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

                med = util.meddistance(tr.stack_xy(), 1000)
                list_gwidth = np.hstack(((med ** 2) * (2.0 ** np.linspace(-4, 4, 20))))
                list_gwidth.sort()
                list_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

                besti, powers = tst.QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha=0.05)
                best_ker = list_kernels[besti]

                mmd_train_test_results[start_point]['perm_mmds1'] = tst.QuadMMDTest.permutation_list_mmd2(xtr, ytr, k,
                                                                                                          n_permute=2000)
                mmd_train_test_results[start_point]['chi2_weights'] = chi2_weights
                mmd_train_test_results[start_point]['sim_mmds'] = sim_mmds
                mmd_train_test_results[start_point]['sig2'] = sig2
                mmd_train_test_results[start_point]['Kxy'] = Kxy
                mmd_train_test_results[start_point]['mean'] = mean
                mmd_train_test_results[start_point]['var'] = var
                mmd_train_test_results[start_point]['Kxx'] = Kx
                mmd_train_test_results[start_point]['Kyy'] = Ky
                mmd_train_test_results[start_point]['mean_gram'] = mean_gram
                mmd_train_test_results[start_point]['var_gram'] = var_gram
                mmd_train_test_results[start_point]['med'] = med
                mmd_train_test_results[start_point]['list_gwidth'] = list_gwidth
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
            # Generate filename and save the results for this symbol and mfdfakey
            filename = f"{symbol}_{mfdfakey}_OBI_results.pkl"
            file_path = os.path.join(
                '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance/OBIMFDFAResults', filename)
            # Save the results
            pickle_out_dict_one = os.path.join(obimfdfaResultsPath, f"{symbol}_{mfdfakey}_OBI_quad_MMD_test.pkl")
            with open(pickle_out_dict_one, 'wb') as file:
                pickle.dump(mmd_train_test_results, file, protocol=pickle.HIGHEST_PROTOCOL)
            return {"symbol": symbol, "mfdfakey": mfdfakey, "status": "completed", "error": None}

    def save_results(result, obimfdfaPath):
        symbol, mfdfakey, shift, window, mmd_train_test_results = result.values()
        filename = f"{symbol}_{mfdfakey}_shift_{shift}_wind_{window}_OBI_results.pkl"
        resultsPath = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance/OBIMFDFAResults'
        file_path = os.path.join(obimfdfaResultsPath, filename)
        with open(file_path, 'wb') as file:
            pickle.dump(mmd_train_test_results, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Initialization
    a_args = [1, 2, 3]  # Example values for a_args
    second_arg = [1, 4, 5, 6, 8, 10]  # Example values for second_arg
    c = list(itertools.product(a_args, second_arg))
    symbols = ['RX1', 'US1', 'XM1', 'YM1', 'DU1', 'FB1', 'FV1', 'G', 'JB1', 'KE1', 'OE1']  # symbols
    # Paths
    obimfdfaPath = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance/OBIMFDFA'
    reconLOBs = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB'
    obimfdfaResultsPath = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance/OBIMFDFAResults'
    figuresLocation = '/home/ak/Documents/Research/Papers/StylisedFactsPaper/figures'
    symbols = sorted(os.listdir(reconLOBs))
    symbolFolders = [os.path.join(reconLOBs, symbol) for symbol in symbols]
    # Keys of interest
    mfdfakeys = ['orderBookImbalance', 'n', 'F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect', 'qs']

    # Initialize a nested defaultdict
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    files = os.listdir(obimfdfaPath)
    files = [f for f in files if '_data' not in f]

    # Iterate over files and aggregate data
    for filename in files:
        symbol = filename.split("_")[0]  # Extracting symbol from filename
        file_path = os.path.join(obimfdfaPath, filename)
        unpickledData = pd.read_pickle(file_path)  # Assuming data is a dict of dicts
        keys = list(unpickledData.keys())

        # Populate data_dict
        for idx in keys:
            for key in mfdfakeys:
                data_dict[symbol][key][idx] = unpickledData[idx].get(key, {})

    # Save the slices of data_dict to files
    save_data_slices(data_dict, obimfdfaPath, mfdfakeys)

    # Parallel execution
    num_processes = max(1, multiprocessing.cpu_count())
    # Create a list of all combinations of symbol, mfdfakey, and other arguments
    all_combinations = [(symbol, mfdfakey, data_dict, obimfdfaPath, a_args, second_arg)
                        for symbol in data_dict.keys()
                        for mfdfakey in mfdfakeys]

    # Create tasks for multiprocessing
    tasks = [(symbol, mfdfakey, os.path.join(obimfdfaPath, f"{symbol}_{mfdfakey}_data_slice.json"),
              obimfdfaPath, a_args, second_arg)
             for symbol in symbols for mfdfakey in mfdfakeys]

    # Run the tasks using multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_data_from_file, tasks)

    # # Create a pool of worker processes
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     # Map the process_symbol_mfdfakey function to all combinations
    #     pool.starmap(process_symbol_mfdfakey, all_combinations)
    #
    # # Close the pool and wait for all tasks to complete.
    pool.close()
    pool.join()
