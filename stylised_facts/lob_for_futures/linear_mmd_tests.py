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
from pathlib import Path
import sys
import time
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle


def save_results_to_file(test_results_one_dict, test_results_two_dict, shift, window, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    test_results_one_file = os.path.join(output_dir, f"test_results_one_shift_{shift}_window_{window}.pkl")
    test_results_two_file = os.path.join(output_dir, f"test_results_two_shift_{shift}_window_{window}.pkl")

    with open(test_results_one_file, 'wb') as f:
        pickle.dump(test_results_one_dict, f)

    with open(test_results_two_file, 'wb') as f:
        pickle.dump(test_results_two_dict, f)


def load_results(output_dir):
    results_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

    results = []
    for file in results_files:
        filepath = os.path.join(output_dir, file)
        df = pd.read_csv(filepath)
        df['filename'] = file
        results.append(df)

    return pd.concat(results, ignore_index=True)


def generate_latex_summary_table(dataframe):
    latex_table = dataframe.to_latex(index=False)
    return latex_table


def analyze_and_save(shift, window, analyzer, output_dir):
    try:
        test_results_one_dict, test_results_two_dict = analyzer.analyze(shift=shift, window=window)
        save_results_to_file(test_results_one_dict, test_results_two_dict, shift, window, output_dir)
    except ValueError as e:
        print(f"Error occurred during analyze for shift {shift} and window {window}:", e)


def load_and_concatenate_results_csv(output_dir):
    output_dir = Path(output_dir)
    csv_files = sorted(output_dir.glob("*.csv"))
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    concatenated_df = pd.concat(dfs, axis=0)
    return concatenated_df


def load_and_concatenate_results(output_dir):
    output_dir = Path(output_dir)
    pkl_files = sorted(output_dir.glob("*.pkl"))
    dfs = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            df = pickle.load(f)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)
    concatenated_df = pd.concat(dfs, axis=0)
    return concatenated_df


def load_and_concatenate_results_default_dict(output_dir):
    output_dir = Path(output_dir)
    pkl_files = sorted(output_dir.glob("*.pkl"))
    dfs = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            results_dict = pickle.load(f)
        for key in results_dict:
            if isinstance(key, int):
                key = (key,)
            results_dict[key]["shift"] = key[0]
            results_dict[key]["window"] = key[1]
        df = pd.DataFrame.from_dict(results_dict, orient="index")
        dfs.append(df)
    concatenated_df = pd.concat(dfs, axis=0)
    return concatenated_df


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


class LinearMMDAnalyzer:
    """
    A class to analyze Linear MMD tests on symbol data and save the results.
    """

    def __init__(self, symbol='', mfdfaDataFrames='',
                 LinearMMDOutputFiles='', bar_choice='', variable=''):
        """
        Initialize the LinearMMDAnalyzer with symbol, input, and output paths.

        :param symbol: str, the symbol to analyze.
        :param mfdfaDataFrames: str, the base directory containing the symbol data.
        :param LinearMMDOutputFiles: str, the base directory where the output files will be saved.
        :param bar_choice: str, the bar choice to process (e.g., 'dollar').
        :param variable: str, the variable to analyze.
        """
        self.symbol = symbol
        self.mfdfaDataFrames = mfdfaDataFrames
        self.LinearMMDOutputFiles = LinearMMDOutputFiles
        self.symbol_mfdfa_Frames_loc = os.path.join(self.mfdfaDataFrames, self.symbol)
        self.bar_choice = bar_choice
        self.variable = variable
        self.symbol_mfdfa_Frames_results_bar_choice = os.path.join(self.symbol_mfdfa_Frames_loc, 'results',
                                                                   self.bar_choice)

    def get_var_files(self):
        return [f for f in os.listdir(self.symbol_mfdfa_Frames_results_bar_choice) if str(self.variable) in f]

    def get_unpickled_file(self):
        """
        the unpickled file containing the symbol/variable data.
            :return: probably default dict.
        """
        for file in list(self.get_var_files()):
            return pd.read_pickle(os.path.join(self.symbol_mfdfa_Frames_results_bar_choice, file))

    def analyze(self, shift=1, window=5, kgauss_instance=None):
        """
        Perform the Linear MMD analysis on the symbol data and return the results.

        :param shift: int, the shift parameter for the analysis.
        :param window: int, the window parameter for the analysis.
        :param kgauss_instance: KGauss, an optional instance of the KGauss class to use for the kernel.
        :return: tuple, a tuple containing the test results dictionaries.
        """
        unpickled_Df = self.get_unpickled_file()
        length = unpickled_Df.shape[1]
        test_results_one_dict = defaultdict(dict)
        test_results_two_dict = defaultdict(dict)
        for start_point in range(0, (length - window - shift)):
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

            if kgauss_instance is None:

                list_kernels = [create_kgauss(w ** 2, default_sigma2=1) for w in widths]

            else:
                list_kernels = [kgauss_instance for _ in range(len(widths))]

            besti, powers = tst.LinearMMDTest.grid_search_kernel(tr, list_kernels, alpha=0.01)

            # The actual test
            best_ker = list_kernels[besti]
            lin_mmd_test = tst.LinearMMDTest(best_ker, alpha=0.01)

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

        return test_results_one_dict, test_results_two_dict

    def analyze_multiple_variables(self, variables, shift=1, window=5):
        """
        Perform the Linear MMD analysis on multiple variables and return the results.

        :param variables: list, a list of variables to analyze.
        :param shift: int, the shift parameter for the analysis.
        :param window: int, the window parameter for the analysis.
        :return: list, a list of dictionaries containing the test results for each variable.
        """
        results_list = []

        for variable in variables:
            self.variable = variable
            test_results_one_dict, test_results_two_dict = self.analyze(shift=shift, window=window)
            results_dict = {
                'variable': variable,
                'test_results_one_dict': test_results_one_dict,
                'test_results_two_dict': test_results_two_dict
            }
            results_list.append(results_dict)

        return results_list

    def save_results(self, test_results_one_dict, test_results_two_dict, shift, window):
        """
        Save the test results to pickle files.

        :param test_results_one_dict: dict, the first test results dictionary.
        :param test_results_two_dict: dict, the second test results dictionary.
        :param shift: int, the shift parameter for the analysis.
        :param window: int, the window parameter for the analysis.
        """
        pickle_out_dict_one = os.path.join(self.LinearMMDOutputFiles, "".join(
            (str(self.symbol) + "_" + str(self.variable) + "_shift_" + str(shift) + "_wind_" + str(window) + "_" + str(
                'linear_test') + "_ONE.pkl")))
        pickle.dump(test_results_one_dict, open(pickle_out_dict_one, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        pickle_out_dict_two = os.path.join(self.LinearMMDOutputFiles, "".join(
            (str(self.symbol) + "_" + str(self.variable) + "_shift_" + str(shift) + "_wind_" + str(window) + "_" + str(
                'linear_test') + "_TWO.pkl")))
        pickle.dump(test_results_two_dict, open(pickle_out_dict_two, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    experimentOne = '/media/ak/T71/August11th2022Experiments/experimentOne'
    mfdfaDataFrames = '/media/ak/T71/August11th2022Experiments/mfdfaDataFrames'
    symbol = 'RX1'
    symbolPath = os.path.join(mfdfaDataFrames, symbol)

    bar_choice = 'dollar'
    variable = 'alpha'
    print(symbolPath, os.path.isdir(symbolPath))

    symbol_barChoicePath = os.path.join(symbolPath, 'results', bar_choice)
    print(symbol_barChoicePath)

    files = [f for f in os.listdir(symbol_barChoicePath) if str(bar_choice) in f]
    variables = ['n_F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect']

    LinearMMDOutputFiles = os.path.join(experimentOne, 'LinearMMDOutputFiles')
    if not os.path.exists(LinearMMDOutputFiles):
        os.makedirs(LinearMMDOutputFiles)

    analyzer = LinearMMDAnalyzer(symbol=symbol, mfdfaDataFrames=mfdfaDataFrames,
                                 LinearMMDOutputFiles=LinearMMDOutputFiles,
                                 bar_choice=bar_choice, variable=variable)
    gaussian_kernel = create_kgauss(sigma2=-1, default_sigma2=1)
    analyzer.analyze(kgauss_instance=gaussian_kernel)
