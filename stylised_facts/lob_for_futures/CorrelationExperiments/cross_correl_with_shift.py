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

    @staticmethod
    def lag_shift_dataframe(data: pd.DataFrame, lag: int, columns_to_keep: list) -> pd.DataFrame:
        """
        Apply a lag shift to a DataFrame, keeping only specified columns, and returning both the original and lagged values.

        Args:
            data (pd.DataFrame): The input dataframe.
            lag (int): The number of periods to shift the data.
            columns_to_keep (list): A list of columns to keep and apply the shift to.

        Returns:
            combined_data (pd.DataFrame): The DataFrame containing the original and lagged values of the specified columns.
        """

        # Keep only the specified columns
        reduced_data = data[columns_to_keep]

        # Shift the remaining columns by the specified lag
        shifted_data = reduced_data.shift(lag)

        # Concatenate the original and shifted data
        combined_data = pd.concat([reduced_data, shifted_data], axis=1)

        # Update column names to indicate original and lagged values
        combined_data.columns = [f"{col}_orig" if i < len(columns_to_keep) else f"{col}_lag_{lag}" for i, col in
                                 enumerate(combined_data.columns)]

        return combined_data
    @staticmethod
    def lag_shift_multiple(data: pd.DataFrame, max_lag: int, columns_to_keep: list) -> pd.DataFrame:
        """
        Apply multiple lag shifts to a DataFrame, keeping only specified columns, and returning both the original and lagged values.

        Args:
            data (pd.DataFrame): The input dataframe.
            max_lag (int): The maximum number of periods to shift the data.
            columns_to_keep (list): A list of columns to keep and apply the shift to.

        Returns:
            combined_data (pd.DataFrame): The DataFrame containing the original and lagged values of the specified columns for all lags up to max_lag.
        """

        # Keep only the specified columns
        reduced_data = data[columns_to_keep]

        # Create a list to store dataframes with different lag values
        lag_dataframes = [reduced_data]

        # Apply lags from 1 to max_lag and append the lagged dataframes to the list
        for lag in range(1, max_lag + 1):
            shifted_data = reduced_data.shift(lag)
            shifted_data.columns = [f"{col}_lag{lag}" for col in shifted_data.columns]
            lag_dataframes.append(shifted_data)

        combined_data = pd.concat(lag_dataframes, axis=1)
        return combined_data


if __name__ == '__main__':
    # Usage
    base_path = "/media/ak/Data1/ExperimentData/August11th2022Experiments/ExperimentInputFiles"
    symbol = "JB1"
    bar = "dollar"
    test_file = "JB1_dollar_20180420_exp_df.pkl"

    select = [0, 1, 3]  # Select files to load by index

    data_loader = DataLoader(base_path, symbol, bar)
    data = data_loader.load_selected_files(select)
    # print(data)
    # Compute the median of the shifted data
    # Apply shift to the data
    df_shifted = data_loader.df_derived_by_shift(data,lag =3, non_derived_cols= ["GK_vol"])
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