import os
import pandas as pd
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '/directory/tothe/handshakefile/')
sys.path.append('/home/ak/Documents/PaperCode/stylised_facts')

mfdfaDataFrames = '/media/ak/T71/August11th2022Experiments/mfdfaDataFrames'  # Input Files
LinearMMDOutputFiles = '/media/ak/T71/August11th2022Experiments/ExperimentOne/TestMMDOutputFiles'  # Location to Save Files
experimentOne = '/media/ak/T71/August11th2022Experiments/ExperimentOne'


def get_filtered_files(path, bar_choice):
    return sorted([f for f in os.listdir(path) if str(bar_choice) in f])


def read_pickle_file(file_path, bar_choice, variable):
    return pd.read_pickle(file_path)[str(bar_choice)][str(variable)]


def create_dataframe_from_dict(unpickled_all_values):
    return pd.DataFrame.from_dict(unpickled_all_values).fillna(0)


def save_dataframe_to_pickle(df, save_path, protocol=pickle.HIGHEST_PROTOCOL):
    with open(save_path, 'wb') as f:
        pickle.dump(df, f, protocol=protocol)
    print(f'saving: {save_path}')


def process_symbol(symbol):
    symbol_analyzer = SymbolAnalyzer(symbol=symbol, experimentOne=experimentOne,
                                     bars=['tick', 'calendar', 'dollar', 'volume'])
    symbol_analyzer.check_directory()
    symbol_analyzer.process_bars(max_workers=4)


def process_symbols(symbols, max_symbol_analyzers):
    """
    Process a list of symbols in parallel.

    :param symbols: list, a list of symbols to process.
    :param max_symbol_analyzers: int, the maximum number of concurrent symbol analyzers.
    """
    with ThreadPoolExecutor(max_workers=max_symbol_analyzers) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]

    for future in futures:
        future.result()


class SymbolAnalyzer:
    """
    A class to analyze symbols using multiple bar choices and save the results.
    """
    def __init__(self, symbol='FB1', experimentOne='', bars=None):
        """
        Initialize the SymbolAnalyzer with a symbol, base directory, and a list of bars.

        :param symbol: str, the symbol to analyze.
        :param experimentOne: str, the base directory containing the symbol data.
        :param bars: list, a list of bar choices to process (e.g., ['tick', 'calendar', 'dollar', 'volume']).
        """
        self.symbol = symbol
        self.symbolPath = os.path.join(experimentOne, str(self.symbol))
        self.resultsMainPath = '/media/ak/T71/August11th2022Experiments/mfdfaDataFrames'
        self.resultsPath = os.path.join(self.resultsMainPath, self.symbol, 'results')
        os.makedirs(self.resultsPath, exist_ok=True)
        self.bars = bars or ['tick']
        self.variables = ['n_F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect']

    def check_directory(self):
        """
        Print the symbol path and a boolean indicating if the directory exists.
        """
        print(self.symbolPath, os.path.isdir(self.symbolPath))

    def get_files(self, bar):
        """
        Get the list of files for a specific bar choice.

        :param bar: str, the bar choice to search for in the filenames.
        :return: list, a list of filenames containing the specific bar choice.
        """
        files = [f for f in os.listdir(self.symbolPath) if str(bar) in f]
        return files

    def process_file(self, file, bar, variable):
        """
        Process a single file for a specific bar and variable.

        :param file: str, the file to process.
        :param bar: str, the bar choice.
        :param variable: str, the variable to extract from the file.
        :return: tuple, the index and variable_array extracted from the file, or (None, None) if an error occurs.
        """
        try:
            file_loc = os.path.join(self.symbolPath, file)
            variable_array = pd.read_pickle(file_loc)[str(bar)][str(variable)]
            index = file.split('_')[1].split('.')[0]
            return index, variable_array
        except TypeError:
            print(f"Error processing file {file} for bar {bar} and variable {variable}.")
            return None, None

    def process_files_parallel(self, files, bar, max_workers=4):
        """
        Process a list of files in parallel for a specific bar choice and save the results.

        :param files: list, a list of files to process.
        :param bar: str, the bar choice.
        :param max_workers: int, the maximum number of concurrent workers to use.
        """
        for variable in self.variables:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = [executor.submit(self.process_file, file, bar, variable) for file in files]
            # result_dict = {r.result()[0]: r.result()[1] for r in results}
            result_dict = {r.result()[0]: r.result()[1] for r in results if r.result()[0] is not None}
            bar_results_path = os.path.join(self.resultsPath, bar)
            os.makedirs(bar_results_path, exist_ok=True)
            save_path = os.path.join(bar_results_path, f"{self.symbol}_{bar}_{variable}.pkl")
            self.save_dataframe_to_pickle(pd.DataFrame(result_dict), save_path)

    def save_dataframe_to_pickle(self, df, save_path, protocol=pickle.HIGHEST_PROTOCOL):
        with open(save_path, 'wb') as f:
            pickle.dump(df, f, protocol=protocol)
        print(f'saving: {save_path}')

    def process_bars(self, max_workers=4):
        """
        Process all the bar choices for a symbol using the SymbolAnalyzer class.

        :param max_workers: int, the maximum number of concurrent workers to use.
        """
        for bar in self.bars:
            files = self.get_files(bar)
            self.process_files_parallel(files, bar, max_workers)


if __name__ == '__main__':
    experimentOne = experimentOne
    symbols = ['TY1', 'RX1', 'XM1', 'FV1', 'XM1', 'KE1', 'JB1', 'TU1', 'US1', 'G_1', 'FB1','DU1' ]
    max_symbol_analyzers = 4

    process_symbols(symbols, max_symbol_analyzers)