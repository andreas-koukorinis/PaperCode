import os
import pickle
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import freqopttest.util as util
import freqopttest.data as data
import freqopttest.kernel as kernel
import freqopttest.tst as tst
import collections
from collections import defaultdict
import freqopttest.glo as glo
import multiprocessing
from multiprocessing import Pool
import sys
import time
import os
import pandas as pd
import pickle
import itertools

# Set plot parameters
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'


def load_data(filename):
    """
      Load and return data from a given pickle file.

      Args:
          filename (str): Path to the pickle file.

      Returns:
          DataFrame: Data loaded from the pickle file.
      """
    return pd.read_pickle(filename)


def process_data(args):
    """
    Process data for a given symbol and mfdfakey. This function includes loading data,
    performing calculations, and preparing results.

    Args:
        args (tuple): A tuple containing the symbol, mfdfakey, filename, a_args, and second_arg.

    Returns:
        list: A list of result dictionaries for each processed dataset.
    """
    symbol, mfdfakey, filename, a_args, second_arg = args
    data_dict = defaultdict(dict)
    pickledata = load_data(filename)

    # Data processing logic (populate data_dict)
    for idx in pickledata:
        for key in mfdfakey:
            data_dict[symbol][key][idx] = pickledata[idx][key]

    choice_dataframe = pd.DataFrame.from_dict(data_dict[symbol][mfdfakey])
    length = choice_dataframe.shape[1]

    results = []
    for shift, window in itertools.product(a_args, second_arg):
        for start_point in range(0, (length - window - shift)):
            end_point = start_point + shift
            # Extract X, Y, Z arrays (replace with actual slicing logic)
            X, Y, Z = np.array([]), np.array([]), np.array([])
            result = {"symbol": symbol, "mfdfakey": mfdfakey, "shift": shift, "window": window, "data": "Sample Result"}
            results.append(result)

    return results
def save_results(result, obim_mdfdfa_path):
    """
    Save the given results to a pickle file.

    Args:
        result (dict): A dictionary containing the results and metadata.
        obim_mdfdfa_path (str): Path to save the output pickle file.
    """
    symbol, mfdfakey, shift, window, data = result.values()
    filename = f"{symbol}_{mfdfakey}_shift_{shift}_wind_{window}_OBI_quad_MMD_test.pkl"
    pickle_out_path = os.path.join(obim_mdfdfa_path, filename)
    with open(pickle_out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
def run_parallel_processing(obi_files, symbols, mfdfa_keys, a_args, second_arg, obim_mdfdfa_path):
    """
    Run the data processing in parallel for each combination of symbols and mfdfa_keys.

    Args:
        obi_files (list): List of paths to data files.
        symbols (list): List of symbols to process.
        mfdfa_keys (list): List of mfdfa keys.
        a_args (list): First set of arguments for processing.
        second_arg (list): Second set of arguments for processing.
        obim_mdfdfa_path (str): Path to save output files.
    """
    pool = multiprocessing.Pool(processes=4)  # Adjust the number of processes as needed
    tasks = [(symbol, mfdfa_keys, filename, a_args, second_arg) for filename in obi_files for symbol in symbols]

    results = pool.map(process_data, tasks)
    pool.close()
    pool.join()

    # Flatten the list of lists
    all_results = [item for sublist in results for item in sublist]

    for result in all_results:
        save_results(result, obim_mdfdfa_path)

# Improved function to dump objects
def pickle_dump_obj(destination_path, filename, obj):
    try:
        with open(os.path.join(destination_path, filename), 'wb') as file:
            pickle.dump(obj, file)
        print('Saved:', os.path.join(destination_path, filename))
    except Exception as e:
        print('Error saving file:', e)
def dump_data(destination_path, filename, obj, format='pickle'):
    path = destination_path
    try:
        if format == 'json':
            with open(path, 'w') as file:
                json.dump(obj, file)
        elif format == 'pickle':
            with open(path, 'wb') as file:
                pickle.dump(obj, file)
        print('Saved:', path)
    except Exception as e:
        print('Error saving file:', e)
# two sample testing functions
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
# Main code
if __name__ == "__main__":
    # Directory paths
    recon_lob_path = '/media/ak/Data1/InterestRateFuturesData/ReconstructedLOB'
    figures_location = '/home/ak/Documents/Research/Papers/StylisedFactsPaper/figures'
    obim_mdfdfa_path = os.path.join(recon_lob_path, 'OrderBookImbalance', 'OBIMFDFA')

    # List of symbols
    symbols = ['DU1', 'FB1', 'FV1', 'G', 'JB1', 'KE1', 'OE1', 'RX1', 'US1', 'XM1', 'YM1']

    # Keys of interest
    mfdfa_keys = ['orderBookImbalance', 'n', 'F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect', 'qs']

    # Collecting file paths
    extremes_path = os.path.join(recon_lob_path, 'OrderBookImbalance', 'extremes')
    obi_files = [os.path.join(obim_mdfdfa_path, f) for f in os.listdir(obim_mdfdfa_path) if f.endswith('mfdfa.pkl')]

    # Initialize a nested defaultdict
    data_dict = defaultdict(lambda: defaultdict(dict))

    # Process files
    for filename in obi_files:
        symbol = filename.split("_")[0]
        print(filename)
        pickledata = pd.read_pickle(os.path.join(obim_mdfdfa_path, filename)) # Assuming data is a dict of dicts
        keys = list(pickledata.keys())
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        for idx in keys:
            for key in mfdfa_keys:
                data_dict[symbol][key][idx] = pickledata[idx][key]

    # Assuming `a_args` and `second_arg` are defined
    mmd_train_test_results = defaultdict(dict)
    a_args = [1, 2, 3]  # Example values
    second_arg = [1, 4, 5, 6, 8, 10]  # Example values

    c = list(itertools.product(a_args, second_arg))
    symbol = 'DU1'
    mfdfakey = 'alpha'
    choice_dataframe = pd.DataFrame.from_dict( data_dict[str(symbol)][str(mfdfakey)] )
    # Replace 'len' with a different variable name, like 'length'
    length = choice_dataframe.shape[1]  # This should be set to your actual length value
    print('######################### LENGTH #########################')
    print(length)
    print('######################### LENGTH #########################')

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
                # Ensure sig2 is positive
                if np.isnan(sig2) or sig2 <= 0:
                    print("Warning: sig2 is not positive. Setting sig2 to a default positive value.")
                    sig2 = 1  # or some other appropriate default value
                else:
                    # Actions to take if sig2 is already positive
                    # For example, logging the value
                    print(f"Using provided sig2 value: {sig2}")
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
            print(mmd_train_test_results)
            pickle_out_dict_one = os.path.join(obim_mdfdfa_path, "".join(
                (str(symbol) + "_" + str(mfdfakey) + "_shift_" + str(shift) + "_wind_" + str(window) + "_" + str(
                    'OBI_quad_MMD_test') + ".pkl")))
            pickle.dump(mmd_train_test_results, open(pickle_out_dict_one, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


