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
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations


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
            list_kernels_verbose = [kernel.KGauss(gw2).__str__() for gw2 in list_gwidth]

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
            mmd_train_test_results[start_point]['list_kernels'] = list_kernels_verbose
            mmd_train_test_results[start_point]['besti'] = besti
            mmd_train_test_results[start_point]['powers'] = powers
            mmd_train_test_results[start_point]['best_ker'] = best_ker.__str__()

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
        """
        Initialize QuadMMDAnalysis with the input data frame and other parameters.

        :param df: Input data frame containing the data to be analyzed.
        :param symbol: A string representing the symbol of the financial instrument.
        :param output_directory: A string representing the path to the output directory.
        :param bar_choice: A string representing the choice of bars (e.g., "BarChoice").
        :param variable: A string representing the variable (e.g., "Variable").
               """

        self.df = df
        self.symbol = symbol
        self.LinearMMDOutputFiles = LinearMMDOutputFiles
        self.bar_choice = bar_choice
        self.variable = variable

    def run_quad_mmd_analysis(self, start_point, end_point, shift, window):
        """
        Run the perform_quad_mmd_tests method with the parameters set as class attributes.

        :return: A dictionary containing the results of the quad MMD analysis.
        """
        # Run perform_quad_mmd_tests with the parameters set as class attributes
        mmd_train_test_results = self.perform_quad_mmd_tests(start_point, end_point, shift, window)

        # Print the results
        print(mmd_train_test_results)

    def run_single_analysis(self, start_point, end_point, shift, window):
        """
        Run the perform_quad_mmd_tests method for a single combination of start_point, end_point, shift, and window.
        run_single_analysis that takes the start_point, end_point, shift, and window as arguments and returns a tuple containing the combination and its corresponding result.

        :param start_point: The starting index of the column pair.
        :param end_point: The ending index of the column pair.
        :param shift: The shift to be applied during the analysis.
        :param window: The window size to be used during the analysis.
        :return: A tuple containing the combination of start_point, end_point, shift, window, and the result.
        """
        self.start_point = start_point
        self.end_point = end_point
        self.shift = shift
        self.window = window
        result = self.run_quad_mmd_analysis(self.start_point, self.end_point, self.shift, self.window)

        return (start_point, end_point, shift, window, result)

    def run_multiple_quad_mmd_analyses(self, windows, shifts, n_jobs=-1):
        """
        Run the perform_quad_mmd_tests method for multiple combinations of shifts and windows with parallelization.

        :param windows: A list of integers representing the window sizes to be used during the analysis.
        :param shifts: A list of integers representing the shifts to be applied during the analysis.
        :param n_jobs: The number of parallel processes to be used during the analysis.
        :return: A dictionary containing the results of the quad MMD analysis for all combinations.
        """
        # Generate all possible combinations of column pairs, window sizes, and shifts
        column_pairs = list(combinations(self.df.columns, 2))

        # Create a list of arguments for the analyze_column function
        args_list = []
        for start_point, end_point in column_pairs:
            for window in windows:
                for shift in shifts:
                    args_list.append((start_point, end_point, shift, window))

        # Run quad MMD analysis for all combinations and store the results in a dictionary
        results_dict = {}
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for args, result in executor.map(lambda args: self.run_single_analysis(*args), args_list):
                results_dict[args] = result

        return results_dict

        # Modify the run_quad_mmd_analysis method to use smaller ranges

    def faster_run_quad_mmd_analysis(self):
        num_columns = self.df.shape[1]

        # Generate all possible combinations of column pairs, window sizes, and shifts with smaller ranges
        windows = range(5, 101, 10)
        shifts = range(1, 6)
        args_list = []
        result_dict = {}
        for start_point, end_point in list(combinations(range(len(self.df.columns)), 2))[:10]:  # Limit to 10 pairs
            for window in windows:
                for shift in shifts:
                    args_list.append((start_point, end_point, shift, window, result_dict))

        # Use multiprocessing to analyze the data in parallel
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(self.analyze_mmd_column, [(start_point, end_point, shift, window, self, {}) for
                                                             start_point, end_point, shift, window in args_list])

        # Combine the results into a single dictionary
        result_dict = {}
        for key, value in results:
            if value:
                result_dict[key] = value

        # Save the results to a pickle file
        output_file = os.path.join(self.LinearMMDOutputFiles,
                                   f"{self.symbol}_{self.bar_choice}_processedQUADMMDresults_{self.variable}.pickle")

        result_list = []
        for key, value in result_dict.items():
            col_pair = f"{self.df.columns[key[0]]} vs {self.df.columns[key[1] + key[2]]}"
            window, shift = key[3], key[2]
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

    def analyze_mmd_column(self, start_point, end_point, shift, window):
        result_dict = {}
        try:
            result = self.run_single_analysis(start_point, end_point, shift, window)
            result_dict[(start_point, end_point, shift, window)] = result
        except ValueError:
            pass
        return result_dict

    def analyze_mmd_column(self, start_point, end_point, shift, window, result_dict):
        try:
            test_results_one, test_results_two = self.run_single_analysis(start_point, end_point, shift, window)
            col1, col2 = self.df.columns[start_point], self.df.columns[end_point + shift]
            result_key = f"{col1} vs {col2}, window={window}, shift={shift}"
            result_dict[(start_point, end_point, shift, window)] = {'Test Results 1': test_results_one,
                                                                    'Test Results 2': test_results_two}
        except ValueError:
            pass

    def analyze_column(self, start_point, end_point, shift, window, mmd_tester):
        try:
            test_results_one, test_results_two = mmd_tester.analyze(start_point, end_point, shift, window)
            col1, col2 = mmd_tester.df.columns[start_point], mmd_tester.df.columns[end_point + shift]
            result_key = f"{col1} vs {col2}, window={window}, shift={shift}"
            return {result_key: {'Test Results 1': test_results_one, 'Test Results 2': test_results_two}}
        except ValueError:
            pass

    def save_results_to_pickle(self, results, file_name):
        output_file = os.path.join(self.output_directory, file_name)

        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        print("Results saved to:", output_file)


def analyze_mmd_column(self, start_point, end_point, shift, window):
    try:
        result = self.run_single_analysis(start_point, end_point, shift, window)
        col1, col2 = self.df.columns[start_point], self.df.columns[end_point + shift]
        result_key = f"{col1} vs {col2}, window={window}, shift={shift}"
        return ((start_point, end_point, shift, window), {'Test Results 1': result[0], 'Test Results 2': result[1]})
    except ValueError:
        pass

def analyze_column(unpickled_df, start_point, end_point, shift, window, mmd_tester):
    try:
        test_results_one, test_results_two = mmd_tester.analyze(start_point, end_point, shift, window)
        col1, col2 = unpickled_df.columns[start_point], unpickled_df.columns[end_point + shift]
        result_key = f"{col1} vs {col2}, window={window}, shift={shift}"
        return {result_key: {'Test Results 1': test_results_one, 'Test Results 2': test_results_two}}
    except ValueError:
        pass


if __name__ == '__main__':
    # need to load a dataframe here
    symbol = 'XM1'
    LinearMMDInputFiles = '/media/ak/T7/August11th2022Experiments/LinearMMDInputFiles/'
    linear_mmd_output_files = '/media/ak/T7/August11th2022Experiments/QuadMMDOutputFiles'
    bar_choice = 'tick'

    outputDir = '/media/ak/T7/August11th2022Experiments/LinearMMDOutputFiles'
    variables = ['n_F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect']
    file = os.path.join(LinearMMDInputFiles,
                        [f for f in os.listdir(LinearMMDInputFiles) if (str(symbol) and str(bar_choice)) in f][0])
    data_dict = pd.read_pickle(file)

    # Create a list of DataFrames with names from the dictionary keys
    dataframes = [pd.DataFrame(data_dict[key]).assign(name=key) for key in data_dict]

    n_f = data_dict['n_F']
    list_H = data_dict['list_H']
    tau_df = pd.DataFrame.from_dict(data_dict['tau'])
    alpha_df = pd.DataFrame.from_dict(data_dict['alpha'])

    # # # this is the processing code!
    unpickled_dataframe = tau_df
    variable = 'tau'  # bar_choice variable (based on above) - this is the mfdfa variable
    output_name = "_".join((str(symbol), str(bar_choice), 'processedQUADMMDresults', str(variable)))
    num_columns = unpickled_dataframe.shape[1]
    quad_mmd_analysis = QuadMMDAnalysis(unpickled_dataframe, symbol, linear_mmd_output_files, bar_choice, variable)
    # start_point = 0
    # end_point = 1
    # shift = 1
    # window = 10
    # quad_mmd_analysis.run_quad_mmd_analysis(start_point, end_point, shift, window)
    windows = [2, ]
    shifts = [1, 2,]
    # quad_mmd_analysis.run_multiple_quad_mmd_analyses(windows, shifts)
    quad_mmd_analysis.faster_run_quad_mmd_analysis()
