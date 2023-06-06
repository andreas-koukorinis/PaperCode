import numpy as np
import os
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
from itertools import combinations
import itertools
from multiprocessing import Pool, cpu_count
import json
import os
import pickle


# functions:

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


class MMDTester:
    def __init__(self, df):
        self.df = df

    def get_data(self, start_point, end_point, shift, window):
        min_length = min(self.df.iloc[:, start_point:end_point].shape[0],
                         self.df.iloc[:, end_point + shift:end_point + 2 * shift].shape[0],
                         self.df.iloc[:, start_point + window:end_point + window].shape[0])

        X = np.array(self.df.iloc[:min_length, start_point:end_point])
        Y = np.array(self.df.iloc[:min_length, end_point + shift:end_point + 2 * shift])
        Z = np.array(self.df.iloc[:min_length, start_point + window:end_point + window])

        return X, Y, Z

    def perform_tests(self, X, Y, Z):
        data_sample = data.TSTData(X, Y)
        test_data_one = data.TSTData(X, Z)
        test_data_two = data.TSTData(Y, Z)

        tr, te = data_sample.split_tr_te(tr_proportion=0.9, seed=100)

        med = util.meddistance(tr.stack_xy())
        widths = [(med * f) for f in 2.0 ** np.linspace(0, 4, 25)]
        try:

            list_kernels = [kernel.KGauss(w ** 2) for w in widths]
            print('using these', list_kernels)
        except AssertionError:
            print('setting sigma2 =1 as sigma2 > 0, must be > 0')
            list_kernels = [create_kgauss(w ** 2, default_sigma2=1) for w in widths]

        besti, powers = tst.LinearMMDTest.grid_search_kernel(tr, list_kernels, alpha=0.01)

        plt.plot(widths, powers, 'o-')
        plt.xlabel('Gaussian width')
        plt.ylabel('test power')
        plt.title('median distance = %.3g. Best width: %.3g' % (med, widths[besti]))
        plt.show()

        best_ker = list_kernels[besti]
        lin_mmd_test = tst.LinearMMDTest(best_ker, alpha=0.01)

        test_results_one = {
            'widths': widths,
            'med': med,
            'besti': besti,
            'powers': powers,
            'med_on_test_data': util.meddistance(test_data_one.stack_xy()),
            'test_result': lin_mmd_test.perform_test(test_data_one),
            'test_variance': lin_mmd_test.variance(X, Z, best_ker),
            'two_moments': lin_mmd_test.two_moments(X, Z, best_ker),
            'compute_unbiased_linear_estimator': lin_mmd_test.compute_stat(test_data_one)
        }

        test_results_two = {
            'test_result': lin_mmd_test.perform_test(test_data_two),
            'test_variance': lin_mmd_test.variance(Y, Z, best_ker),
            'med_on_test_data': util.meddistance(test_data_two.stack_xy()),
            'two_moments': lin_mmd_test.two_moments(Y, Z, best_ker),
            'compute_unbiased_linear_estimator': lin_mmd_test.compute_stat(test_data_two)
        }

        return test_results_one, test_results_two



    def perform_quad_mmd_tests(self, start_point, end_point, shift, window):


        X, Y, Z = self.get_data(start_point, end_point, shift, window)

        # Initialize a dictionary to store the results
        mmd_train_test_results = defaultdict(dict)

        try:
            tr_data = data.TSTData(X, Y)
            test_data_one = data.TSTData(X, Z)
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

            # Save the results in the mmd_train_test_results dictionary
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

            return mmd_train_test_results

    def analyze(self, start_point, end_point, shift, window):
        X, Y, Z = self.get_data(start_point, end_point, shift, window)
        return self.perform_tests(X, Y, Z)


def analyze_column(mmd_tester, unpickled_df, start_point, end_point, shift, window):
    try:
        test_results_one, test_results_two = mmd_tester.analyze(start_point, end_point, shift, window)
        col1, col2 = unpickled_df.columns[start_point], unpickled_df.columns[end_point + shift]
        result_key = f"{col1} vs {col2}, window={window}, shift={shift}"
        return {result_key: (test_results_one, test_results_two)}
    except ValueError:
        pass


class QuadMMDAnalysis(MMDTester):
    def __init__(self, df, symbol, LinearMMDOutputFiles, bar_choice, variable):
        super().__init__(df)
        self.symbol = symbol
        self.LinearMMDOutputFiles = LinearMMDOutputFiles
        self.bar_choice = bar_choice
        self.variable = variable

    def analyze_column(self, start_point, end_point, shift, window):
        try:
            test_results_one, test_results_two = self.analyze(start_point, end_point, shift, window)
            col1, col2 = self.df.columns[start_point], self.df.columns[end_point + shift]
            result_key = f"{col1} vs {col2}, window={window}, shift={shift}"
            return {result_key: {'Test Results 1': test_results_one, 'Test Results 2': test_results_two}}
        except ValueError:
            pass

    def run_quad_mmd_analysis(self):
        num_columns = self.df.shape[1]

        # Generate all possible combinations of column pairs, window sizes, and shifts
        windows = range(5, 201, 5)
        shifts = range(1, 11)
        args_list = []
        for start_point, end_point in combinations(range(len(self.df.columns)), 2):
            for window in windows:
                for shift in shifts:
                    args_list.append((start_point, end_point, shift, window))

        # Use multiprocessing to analyze the data in parallel
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(self.analyze_column,
                                   [(arg[0], arg[1], arg[2], arg[3]) for arg in args_list])

        # Combine the results into a single dictionary
        result_dict_nested = {}
        for r in results:
            if r:
                result_dict_nested.update(r)

        # Save the results to a pickle file
        output_file = os.path.join(self.LinearMMDOutputFiles,
                                   f"{self.symbol}_{self.bar_choice}_processedLinearMMDresults_{self.variable}.pickle")

        result_list = []
        for key, value in result_dict_nested.items():
            col_pair, window, shift = key.split(', ')
            result_list.append({
                "Column Pair": col_pair,
                "Window": window,
                "Shift": shift,
                "Test Results 1": value["Test Results 1"],
                "Test Results 2": value["Test Results 2"],
            })

        with open(output_file, "wb") as f:
            pickle.dump(result_list, f)

        print("Results saved to:", output_file)


if __name__ == '__main__':
    # need to load a dataframe here
    symbol = 'XM1'
    LinearMMDInputFiles = '/media/ak/T7/August11th2022Experiments/LinearMMDInputFiles/'
    bar_choice = 'tick'
    file = os.path.join(LinearMMDInputFiles,
                        [f for f in os.listdir(LinearMMDInputFiles) if (str(symbol) and str(bar_choice)) in f][0])
    outputDir = '/media/ak/T7/August11th2022Experiments/LinearMMDOutputFiles'
    variables = ['n_F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect']
    data_dict = pd.read_pickle(file)

    # Create a list of DataFrames with names from the dictionary keys
    dataframes = [pd.DataFrame(data_dict[key]).assign(name=key) for key in data_dict]

    n_f = data_dict['n_F']
    list_H = data_dict['list_H']
    tau_df = pd.DataFrame.from_dict(data_dict['tau'])
    alpha_df = pd.DataFrame.from_dict(data_dict['alpha'])

    # # # this is the processing code!
    unpickled_dataframe = tau_df
    mfdfa_var = 'tau'
    output_name = "_".join((str(symbol), str(bar_choice), 'processedLinearMMDresults', str(mfdfa_var)))
    num_columns = unpickled_dataframe.shape[1]

    # Define MMDTester object
    mmd_tester = MMDTester(unpickled_dataframe)


    def analyze_column(unpickled_df, start_point, end_point, shift, window, mmd_tester):
        try:
            test_results_one, test_results_two = mmd_tester.analyze(start_point, end_point, shift, window)
            col1, col2 = unpickled_df.columns[start_point], unpickled_df.columns[end_point + shift]
            result_key = f"{col1} vs {col2}, window={window}, shift={shift}"
            return {result_key: {'Test Results 1': test_results_one, 'Test Results 2': test_results_two}}
        except ValueError:
            pass


    # Generate all possible combinations of column pairs, window sizes, and shifts
    column_pairs = list(combinations(unpickled_dataframe.columns, 2))
    windows = range(5, 201, 5)
    shifts = range(1, 11)

    # Create a list of arguments for the analyze_column function
    args_list = []
    for start_point, end_point in combinations(range(len(unpickled_dataframe.columns)), 2):
        for window in windows:
            for shift in shifts:
                args_list.append((start_point, end_point, shift, window))

    # Create an instance of the MMDTester class outside the analyze_column function
    mmd_tester = MMDTester(unpickled_dataframe)

    # Use multiprocessing to analyze the data in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(analyze_column,
                               [(unpickled_dataframe, arg[0], arg[1], arg[2], arg[3], mmd_tester) for arg in args_list])

    # Combine the results into a single dictionary
    result_dict_nested = {}
    for r in results:
        if r:
            result_dict_nested.update(r)

    # Save the results to a pickle file
    output_file = os.path.join(outputDir, str(output_name) + ".pickle")

    result_list = []
    for key, value in result_dict_nested.items():
        col_pair, window, shift = key.split(', ')
        result_list.append({
            "Column Pair": col_pair,
            "Window": window,
            "Shift": shift,
            "Test Results 1": value["Test Results 1"],
            "Test Results 2": value["Test Results 2"],
        })

    with open(output_file, "wb") as f:
        pickle.dump(result_list, f)

    print("Results saved to:", output_file)