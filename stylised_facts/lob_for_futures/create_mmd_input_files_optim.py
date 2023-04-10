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


class SymbolAnalyzer:
    """
   A class to analyze symbols using multiple bar choices and save the results.
   """

    def __init__(self, symbol='FB1', experimentOne='', bars=None):
        """
       Initialize the SymbolAnalyzer with a symbol, base directory, and a list of bars.

       :param symbol: str, the symbol to analyze.
       :param experimentOne: str, the base directory containing the symbol data.
       :param bars: list, a list of bar choices to process (e.g., ['tick', '1min', '5min']).
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

    def save_dataframe_to_pickle(self, df, save_path, protocol=pickle.HIGHEST_PROTOCOL):
        """
        Save a DataFrame to a pickle file.

        :param df: DataFrame, the DataFrame to save.
        :param save_path: str, the path where the pickle file will be saved.
        :param protocol: int, the pickle protocol to use when saving the DataFrame.
        """
        with open(save_path, 'wb') as f:
            pickle.dump(df, f, protocol=protocol)
        print(f'saving: {save_path}')

    def process_file(self, file, bar, variable, result_dict):
        file_loc = os.path.join(self.symbolPath, file)
        variable_array = pd.read_pickle(file_loc)[str(bar)][str(variable)]
        index = file.split('_')[1].split('.')[0]
        result_dict[index] = variable_array

    def save_results(self, result_dict, bar, variable):
        bar_results_path = os.path.join(self.resultsPath, bar)
        os.makedirs(bar_results_path, exist_ok=True)
        save_path = os.path.join(bar_results_path, f"{self.symbol}_{bar}_{variable}.pkl")
        self.save_dataframe_to_pickle(pd.DataFrame(result_dict), save_path)

    def process_files_parallel(self, files, bar, max_workers=4):
        for variable in self.variables:
            result_dict = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_file, file, bar, variable, result_dict) for file in files]
            self.save_results(result_dict, bar, variable)

    def process_bars(self, max_workers=4):
        for bar in self.bars:
            files = self.get_files(bar)
            self.process_files_parallel(files, bar, max_workers)


from concurrent.futures import ThreadPoolExecutor

if __name__ == '__main__':
    # ...

    # List of symbols to process
    symbols = ['TY1', 'FB1', 'XM1', 'US1',
               'G_1', 'RX1','TU1', 'XM1', 'FV1', 'YM1', 'DU1', 'JB1']

    # Maximum number of concurrent symbol analyzers
    max_symbol_analyzers = 4

    # Process symbols concurrently
    with ThreadPoolExecutor(max_workers=max_symbol_analyzers) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]

    # Wait for all futures to complete
    for future in futures:
        future.result()

