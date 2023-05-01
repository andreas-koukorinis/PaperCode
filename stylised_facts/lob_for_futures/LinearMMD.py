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
            print('using these',list_kernels)
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


if __name__ == '__main__':
    # need to load a dataframe here
    symbol = 'G_1'
    LinearMMDInputFiles = '/media/ak/T7/August11th2022Experiments/LinearMMDInputFiles/'
    bar_choice = 'volume'
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

    ### this is the processing code!
    unpickled_dataframe = alpha_df
    output_name = "_".join((str(symbol), str(bar_choice), 'processedLinearMMDresults', 'alpha.csv'))
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

    # Save the results to a CSV file
    import csv

    with open(os.path.join(outputDir, str(output_name)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Column Pair', 'Window', 'Shift', 'Test Results 1', 'Test Results 2'])
        for key, value in result_dict_nested.items():
            col_pair, window, shift = key.split(', ')
            print('saved')
            writer.writerow([col_pair, window, shift, value['Test Results 1'], value['Test Results 2']])
