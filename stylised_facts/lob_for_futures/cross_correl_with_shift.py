import os
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class DataLoader:
    def __init__(self, base_path, symbol, bar, select=None):
        self.base_path = base_path
        self.symbol = symbol
        self.bar = bar
        self.select = select
        self.data = self.load_data()

    def load_single_file(self, file_path, symbol_path):
        with open(file_path, 'rb') as f:
            df = pickle.load(f)

        variables_to_keep = [
            'price_imbalance', 'weighted_activity_spread', 'GK_vol', 'arrival_rates',
            'returns_normalised', 'median_traded_volume'
        ]
        df = df[variables_to_keep]

        return os.path.basename(file_path), df

    def load_data(self):
        symbol_path = os.path.join(self.base_path, self.symbol)
        file_names = sorted(os.listdir(symbol_path))

        def load_single_file_partial(file_name):
            file_path = os.path.join(symbol_path, file_name)
            return self.load_single_file(file_path)

        if self.select == "single":
            index = 0
            file_name = file_names[index]
            return self.load_single_file(os.path.join(symbol_path, file_name))

        if isinstance(self.select, list):
            selected_files = [file_names[i] for i in self.select]
        else:
            selected_files = [file_name for file_name in file_names if
                              file_name.startswith(f"{self.symbol}_{self.bar}")]

        with ProcessPoolExecutor() as executor:
            data = list(executor.map(load_single_file_partial, selected_files))

        return pd.concat(data, axis=0).reset_index(drop=True)

    def df_derived_by_shift(self, df, lag=0, non_der=[]):
        """
        Shifts columns in a DataFrame by a specified lag, except for columns listed in non_der.

        :param df: DataFrame to shift
        :type df: pd.DataFrame
        :param lag: Number of lags (shifts) to apply, defaults to 0
        :type lag: int, optional
        :param non_der: List of columns to exclude from shifting, defaults to []
        :type non_der: list, optional
        :return: DataFrame with shifted columns
        :rtype: pd.DataFrame
        """
        if not lag:
            return df

        cols_to_shift = df.columns.difference(non_der)
        shifted_dfs = [df[non_der]] + [df[cols_to_shift].shift(periods=i).add_suffix(f'_{i}') for i in
                                       range(1, lag + 1)]

        return pd.concat(shifted_dfs, axis=1)


if __name__ == '__main__':
    pass
# Usage
base_path = "/media/ak/Data1/ExperimentData/August11th2022Experiments/ExperimentInputFiles"
symbol = "JB1"
bar = "dollar"
test_file = "JB1_dollar_20180420_exp_df.pkl"

# Initialize the DataLoader object with the path to the directory containing the data, the symbol (e.g. JB1), and the bar (e.g. dollar)
data_loader = DataLoader(base_path=base_path, symbol='JB1', bar='dollar')

# Load all data files for this symbol and bar
data = data_loader.data

# # Load a single data file
# data_loader_single_file = DataLoader(base_path=base_path, symbol='JB1', bar='dollar', select='single')
# single_file_data = data_loader_single_file.data

# Load a list of data files by specifying their indices in the list of available files
data_loader_file_list = DataLoader(base_path=base_path, symbol='JB1', bar='dollar', select=[0, 2, 4])
file_list_data = data_loader_file_list.data

# Apply a shift to the data, specifying the lag and a list of columns to exclude from shifting
shifted_data = data_loader.df_derived_by_shift(df=data, lag=3, non_der=['GK_vol'])