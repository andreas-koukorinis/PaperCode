import os
import time
import pickle
import pandas as pd
import numpy as np
from fathon import fathonUtils as utils
import fathon
from multiprocessing import Pool
from scipy.stats import linregress
data_folder = 'path/to/data/folder'
symbol_list = sorted(['FB1', 'JB1', 'XM1', 'TY1', 'TU1', 'RX1', 'YM1', 'US1', 'DU1'])  # symbols for T71 USB Drive
window_sizes = utils.linRangeByStep(10, 1000, step=20)
polynomial_order = 2

num_processes = 4  # Set the number of processes for parallelization


class CrossCorrelation:
    def __init__(self, path, symbols, symbol_index, bar_choice):
        """
        Initialize CrossCorrelation object.

        :param path: str, path to the data folder
        :param symbols: list, list of symbols
        :param symbol_index: int, index for the symbol
        :param bar_choice: str, bar for information clock
        """
        self._symbol_index = symbol_index
        self._symbols = symbols
        self._bar = bar_choice
        self._symbol = self._symbols[self._symbol_index]
        self._symbol_filepath = os.path.join(path, str(self._symbol))
        self._list_of_files = os.listdir(self._symbol_filepath)

    def get_data_and_path(self):
        """
        Get list of files and path to those files.

        :return: tuple, (list of files, path to those files)
        """
        files_with_bar = [file for file in self._list_of_files if str(self._bar) in file]
        return files_with_bar, self._symbol_filepath

    def get_all_data_from_file(self, file_index):
        """
        Get all data from file.

        :param file_index: int, index of file in the list
        :return: DataFrame, data from the file
        """
        files_with_bar, symbol_filepath = self.get_data_and_path()
        file_to_get = os.path.join(symbol_filepath, files_with_bar[file_index])
        return pd.read_pickle(file_to_get)

    def get_microvar_data(self, file_index, variable):
        """
        Get microvar data.

        :param file_index: int, index of file in the list
        :param variable: str, micro-structure market variable
        :return: DataFrame, microstructure variable data
        """
        data_dict = self.get_all_data_from_file(file_index)
        return data_dict[str(variable)]

    @staticmethod
    def compute_n_rho(var_a, var_b, index, window_sizes, polynomial_order):
        """
        Compute n and Rho for the given variables.

        :param var_a: DataFrame, first variable for the dCCa rho
        :param var_b: DataFrame, second variable for the dCCa rho
        :param index: int, index for which "day to pick"
        :param window_sizes: list, window sizes used for the computation
        :param polynomial_order: int, polynomial order
        :return: tuple, (n, Rho)
        """
        aggregated_var_a = utils.toAggregated(var_a[index])
        aggregated_var_b = utils.toAggregated(var_b[index])
        try:
            dcca = fathon.DCCA(aggregated_var_a, aggregated_var_b)
            n, _ = dcca.computeFlucVec(window_sizes, polOrd=polynomial_order)
            _, rho = dcca.computeRho(window_sizes, polOrd=polynomial_order)
        except ZeroDivisionError:
            pass
        return n, rho

    @staticmethod
    def compute_n_rho_parallel(args):
        """
        Compute n and Rho for the given variables in parallel.

        :param args: tuple, contains var_a, var_b, index, window_sizes, and polynomial_order
        :return: tuple, (n, Rho)
        """
        var_a, var_b, index, window_sizes, polynomial_order = args
        aggregated_var_a = utils.toAggregated(var_a[index])
        aggregated_var_b = utils.toAggregated(var_b[index])
        try:
            dcca = fathon.DCCA(aggregated_var_a, aggregated_var_b)
            n, _ = dcca.computeFlucVec(window_sizes, polOrd=polynomial_order)
            _, rho = dcca.computeRho(window_sizes, polOrd=polynomial_order)
        except ZeroDivisionError:
            pass
        return n, rho

    def parallel_compute_n_rho(self, var_a, var_b, window_sizes, polynomial_order):
        """
        Parallel computation of n and Rho for a range of indices.

        :param var_a: DataFrame, first variable for the dCCa rho
        :param var_b: DataFrame, second variable for the dCCa rho
        :param window_sizes: list, window sizes used for the computation
        :param polynomial_order: int, polynomial order
        :return: list of tuples, each tuple contains (n, Rho) for a specific index
        """
        num_indices = len(var_a)

        with Pool(num_processes) as pool:
            results = pool.map(self.compute_n_rho_parallel,
                               [(var_a, var_b, index, window_sizes, polynomial_order) for index in range(num_indices)])

        return results

    @staticmethod
    def compute_h_h_intercept_dcca(n, rho):
        """
        Compute the Hurst exponent and the intercept of the linear regression using the DCCA method.

        :param n: array, window sizes
        :param rho: array, rho values corresponding to the window sizes
        :return: tuple, (hurst_exponent, intercept)
        """
        log_n = np.log(n)
        log_rho = np.log(rho)
        hurst_exponent, intercept, _, _, _ = linregress(log_n, log_rho)
        return hurst_exponent, intercept

    def parallel_compute_h_h_intercept_dcca(self, results):
        """
        Parallel computation of Hurst exponent and the intercept for a range of n and Rho values.

        :param results: list of tuples, each tuple contains (n, Rho) for a specific index
        :return: list of tuples, each tuple contains (hurst_exponent, intercept) for a specific index
        """
        with Pool(num_processes) as pool:
            hurst_intercepts = pool.map(self.compute_h_h_intercept_dcca, results)

        return hurst_intercepts


