import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class DataLoader:
    def __init__(self, base_dir, symbol, bar):
        self.base_dir = base_dir
        self.symbol = symbol
        self.bar = bar

    def load_single_file(self, file_path):
        df = pd.read_pickle(file_path)
        return df

    # def load_selected_files(self, file_indices):
    #     symbol_dir = os.path.join(self.base_dir, self.symbol)
    #     all_files = os.listdir(symbol_dir)
    #     selected_files = []
    #     for idx in file_indices:
    #         if f"{self.bar}" in all_files[idx]:
    #             selected_files.append(os.path.join(symbol_dir, all_files[idx]))
    #     with ThreadPoolExecutor() as executor:
    #         data = list(executor.map(self.load_single_file, selected_files))
#     return pd.concat(data)
    def load_selected_files(self, file_indices):
        """
                    Load selected data files based on the indices provided.

                    :param indices: List of indices corresponding to the files to load
                    :type indices: list
                    :return: DataFrame containing data from selected files
                    :rtype: pd.DataFrame
                    """
        symbol_dir = os.path.join(self.base_dir, self.symbol)
        all_files = os.listdir(symbol_dir)
        selected_files = []
        for idx in file_indices:
            if f"{self.bar}" in all_files[idx]:
                selected_files.append(os.path.join(symbol_dir, all_files[idx]))
        data = [self.load_single_file(file) for file in selected_files]
        return data

    def load_all_files(self):
        symbol_dir = os.path.join(self.base_dir, self.symbol)
        all_files = os.listdir(symbol_dir)
        selected_files = [os.path.join(symbol_dir, f) for f in all_files if f"{self.bar}" in f]
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(self.load_single_file, selected_files))
        return pd.concat(data)

    def df_derived_by_shift(df, lag=1, usecols=None, non_derived_cols=None):
        """
        Applies a lag shift to the specified columns of the input dataframe.

        Args:
        - df (pd.DataFrame): the input dataframe
        - lag (int): the number of periods to lag the data by (default: 1)
        - usecols (list): a list of columns to apply the shift to (default: None)
        - non_derived_cols (list): a list of columns that should not be shifted (default: None)

        Returns:
        - pd.DataFrame: a copy of the input dataframe with the specified columns shifted by the specified lag

        """
        if usecols is None:
            usecols = df.columns
        if non_derived_cols is None:
            non_derived_cols = []
        cols_to_shift = [col for col in usecols if col not in non_derived_cols]
        shifted_dataframes = []
        if isinstance(df, pd.DataFrame):
            df = [df]
        for data in df:
            data_shifted = data.copy()
            for col in cols_to_shift:
                data_shifted[col] = data_shifted[col].shift(periods=lag)
            shifted_dataframes.append(data_shifted)
        if len(shifted_dataframes) == 1:
            return shifted_dataframes[0]
        else:
            return shifted_dataframes
    def compute_median(self, file_indices, lag=0, non_der=[], max_workers=4):
        """
        Computes the median of df_derived_by_shift across selected files.

        :param file_indices: Indices of the files to load and compute median
        :type file_indices: list
        :param lag: Number of lags (shifts) to apply, defaults to 0
        :type lag: int, optional
        :param non_der: List of columns to exclude from shifting, defaults to []
        :type non_der: list, optional
        :param max_workers: Maximum number of worker threads to use, defaults to 4
        :type max_workers: int, optional
        :return: Median of df_derived_by_shift across selected files
        :rtype: pd.DataFrame
        """
        data = self.load_selected_files(file_indices)
        shifted_data = [self.df_derived_by_shift(df, lag=lag, non_der=non_der) for df in data]
        median_df = pd.concat(shifted_data).median(level=0)
        return median_df


if __name__ == '__main__':
    # Usage
    base_path = "/media/ak/Data1/ExperimentData/August11th2022Experiments/ExperimentInputFiles"
    symbol = "JB1"
    bar = "dollar"
    test_file = "JB1_dollar_20180420_exp_df.pkl"

    select = [0, 1, 3]  # Select files to load by index

    data_loader = DataLoader(base_path, symbol, bar)
    data = data_loader.load_selected_files(select)
    print(data)
    # Compute the median of the shifted data
    # Apply shift to the data
    df_shifted = data_loader.df_derived_by_shift(data, lag=3, non_der=["GK_vol"])
    median_df = data_loader.compute_median(df_shifted)

    # Print the median DataFrame
    print(median_df)

    #
    # # Apply shift to the data
    # df_shifted = data_loader.df_derived_by_shift(data, lag=3, non_der=["GK_vol"])
    #
    # # Compute median of shifted data across all files
    # median_df = data_loader.compute_median(df_shifted)
    #
    # print(median_df)