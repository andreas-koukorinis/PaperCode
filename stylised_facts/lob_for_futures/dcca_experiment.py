import os
import time
import pickle
import pandas as pd
import numpy as np
import fathon
from fathon import fathonUtils as utils
from multiprocessing import Pool
from scipy.stats import linregress
data_folder = 'path/to/data/folder'
symbol_list = sorted(['FB1', 'JB1', 'XM1', 'TY1', 'TU1', 'RX1', 'YM1', 'G_1', 'KE1', 'FV1', 'US1', 'DU1'])  # symbols for T71 USB Drive
window_sizes = utils.linRangeByStep(10, 1000, step=20)
polynomial_order = 2

num_processes = 4  # Set the number of processes for parallelizatio

# Define paths and folders
# dataFolder = lobFut.dataFolder
# t7 = lobFut.t7folder
# expInputFiles = os.path.join(lobFut.augExpertiments, 'ExperimentInputFiles')
# HiLoData = os.path.join(lobFut.augExpertiments, 'HiLoData')
symbols = ['RX1', 'FB1', 'JB1', 'G_1', 'FV1', 'TY1', 'TU1', 'DU1', 'YM1', 'XM1', 'US1', 'OE1', 'KE1']
t7DataFolder = os.path.join(t7, 'August11th2022Experiments/ExperimentOne/')
laptopDataFolder = os.path.join('/media/ak/OS', 'Data')
expFiles = os.path.join(laptopDataFolder, 'August11th2022Experiments')
destination = os.path.join(laptopDataFolder, 'AthensSoloTripFeb2023')
figuresDestination = '/home/ak/Documents/Papers/StylisedFactsPaper/figures/'
#
# plt.style.use(os.path.join('/home/ak/.config/matplotlib', 'latexstyle3.mplstyle'))
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
    def compute_n_rho(var_a, var_b, window_sizes, polynomial_order):
        """
        Compute n and Rho for the given variables.

        :param var_a: Series, first variable for the dCCa rho
        :param var_b: Series, second variable for the dCCa rho
        :param window_sizes: list, window sizes used for the computation
        :param polynomial_order: int, polynomial order
        :return: tuple, (n, Rho)
        """

        aggregated_var_a = utils.toAggregated(var_a)
        aggregated_var_b = utils.toAggregated(var_b)
        # Check if the last value in window_sizes is smaller than the input vector lengths
        min_input_length = min(len(aggregated_var_a), len(aggregated_var_b))
        if window_sizes[-1] >= min_input_length:
            # Update the window_sizes array
            window_sizes = [size for size in window_sizes if size < min_input_length]
            window_sizes = np.asanyarray([int(size) for size in window_sizes],dtype=np.int64)
        try:
            dcca = fathon.DCCA(aggregated_var_a, aggregated_var_b)
            n, _ = dcca.computeFlucVec(window_sizes, polOrd=polynomial_order)
            _, rho = dcca.computeRho(window_sizes, polOrd=polynomial_order)
        except ZeroDivisionError:
            pass
        return n, rho
    @staticmethod
    def compute_n_F_h_h_intercept_dcca(var_a, var_b, window_sizes, polynomial_order):
        """
        Compute the Hurst exponent and the intercept of the linear regression using the DCCA method.

        :param n: array, window sizes
        :param rho: array, rho values corresponding to the window sizes
        :return: tuple, (hurst_exponent, intercept)
        """
        aggregated_var_a = utils.toAggregated(var_a)
        aggregated_var_b = utils.toAggregated(var_b)
        # Check if the last value in window_sizes is smaller than the input vector lengths
        min_input_length = min(len(aggregated_var_a), len(aggregated_var_b))
        if window_sizes[-1] >= min_input_length:
            # Update the window_sizes array
            window_sizes = [size for size in window_sizes if size < min_input_length]
            window_sizes = np.asanyarray([int(size) for size in window_sizes],dtype=np.int64)
        try:
            dcca = fathon.DCCA(aggregated_var_a, aggregated_var_b)
            n, F = dcca.computeFlucVec(window_sizes, polOrd=polynomial_order)
            H, H_intercept = dcca.fitFlucVec()
        except ZeroDivisionError:
            pass
        return n, F, H, H_intercept


if __name__ == '__main__':
    pass