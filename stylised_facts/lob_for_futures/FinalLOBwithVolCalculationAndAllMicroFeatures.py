import os
import pandas as pd
from typing import List
import json as json
from multiprocessing import Pool
import warnings
import numpy as np

# Ignore the warning
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_pkl_idx(file_loc, file_idx):
    """
    Reads a pickle file based on the given file location and index.
    This function reads a pickle file based on the given file location and index.
    It can be used to read a specific pickle file containing financial data for further analysis.

    :param file_loc: str, file location containing the pickle files
    :param file_idx: int, index of the file to read
    :return: dict, content of the pickle file
    """
    files = os.listdir(file_loc)
    file_idx_loc = os.path.join(file_loc, files[file_idx])
    dict_idx = pd.read_pickle(file_idx_loc)
    return dict_idx


class VolatilityCalculator:
    def __init__(self, rolling_window: int = 5):
        self.rolling_window = rolling_window

    def garman_klass_parkinson_vols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Garman-Klass and Parkinson volatilities for a given DataFrame.

        Parameters:
        df: pd.DataFrame
            Input DataFrame containing the necessary columns: 'MicroPrice_open', 'MicroPrice_high',
            'MicroPrice_low', 'MicroPrice_close'.

        Returns:
        pd.DataFrame
            DataFrame with additional columns 'garman_klass_vol', 'parkinson_vol', and 'bipower_var'.
        """

        df['log_hl'] = (df['MicroPrice_high'].rolling(2).mean() / df['MicroPrice_low'].rolling(2).mean()).replace(
            [np.inf, -np.inf], np.nan).apply(
            np.log).fillna(0)  # the rolling window is 2 so we can normalise a bit the data!
        df['log_cc'] = (df['MicroPrice_close'].rolling(2).mean() / df['MicroPrice_open'].rolling(2).mean()).replace(
            [np.inf, -np.inf], np.nan).apply(
            np.log).fillna(0)
        df['garman_klass_vol'] = np.sqrt(0.5 * df['log_hl'] ** 2 - (2 * np.log(2) - 1) * df['log_cc'] ** 2)
        df['log_hl_squared'] = np.log(
            df['MicroPrice_high'].rolling(2).mean() / df['MicroPrice_low'].rolling(2).mean()) ** 2
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * df['log_hl_squared'].mean())
        df['bipower_var'] = (np.abs(df['log_hl']) * np.abs(df['log_hl'].shift(1))).rolling(self.rolling_window).sum()

        return df

    def calculate_midquote_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Midquote Squared Return (MQSR) and Midquote Absolute Return (MQAR) for a given DataFrame.

        Parameters:
        df: pd.DataFrame
            Input DataFrame containing the necessary columns: 'MicroPrice_open', 'MicroPrice_high',
            'MicroPrice_low', 'MicroPrice_close'.

        Returns:
        pd.DataFrame
            DataFrame with additional columns 'MQSR' (Midquote Squared Return) and 'MQAR' (Midquote Absolute Return).
        """

        midquote = (df['MicroPrice_high'] + df['MicroPrice_low']) / 2

        mqsr = np.square(np.log(midquote) - np.log(midquote.shift(1)))
        mqar = np.abs(np.log(midquote) - np.log(midquote.shift(1)))
        mret = np.log(midquote) - np.log(midquote.shift(1))

        df['MQSR'] = mqsr
        df['MQAR'] = mqar
        df['mret'] = mret

        return df

    def compute_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Effective Spread and Realized Spread for a given DataFrame.

        Parameters:
        df: pd.DataFrame
            Input DataFrame containing the necessary columns.

        Returns:
        pd.DataFrame
            DataFrame with additional columns 'EffectiveSpread' and 'RealizedSpread'.

        Notes:
        - The function assumes that the DataFrame has columns 'BASpread_median', 'time_open', 'time_close',
          'TradePrice_median', and the necessary columns for calculating dt.
        - The function also replaces extreme spread values with the median of non-extreme values.
        """

        df['bid_offer'] = df['BestAsk_median'].rolling(2).mean() - df['BestBid_median'].rolling(2).mean()
        bid_offer_shifted = (df['bid_offer'].shift(-100) + df['bid_offer'].shift(-100)) / 2
        dt = pd.to_timedelta(df['time_close'] - df['time_open']).dt.total_seconds() / 3600

        effective_spread = 2 * dt * (df['TradePrice_median'].rolling(2).mean() - df['bid_offer'])
        realized_spread = 2 * dt * (df['TradePrice_median'].rolling(2).mean() - bid_offer_shifted)

        spreads_df = pd.DataFrame({
            'EffectiveSpread': effective_spread,
            'RealizedSpread': realized_spread
        })
        # Compute the median of the non-extreme values
        lower_percentile = np.percentile(spreads_df, 10)
        upper_percentile = np.percentile(spreads_df, 90)
        non_extreme_spreads = spreads_df[(spreads_df >= lower_percentile) & (spreads_df <= upper_percentile)]
        median_spread = np.median(non_extreme_spreads)

        # Replace extreme values with the median
        spreads_df[spreads_df < lower_percentile] = median_spread
        spreads_df[spreads_df > upper_percentile] = median_spread
        df['EffectiveSpread'] = spreads_df['EffectiveSpread']
        df['RealizedSpread'] = spreads_df['RealizedSpread']

        return df


def resample_event_clock_two(df, event_clock_column: str, event_clock_threshold: int,
                             exclude_median_columns: List[str], columns_to_resample: List[str]) -> pd.DataFrame:
    """
    Resamples a dataframe using an event-based clock.

    Parameters:
    df: pd.DataFrame
        The input dataframe to be resampled.
    event_clock_column: str
        The column name in df to be used as event clock.
    event_clock_threshold: int
        The threshold of cumulative number of events to define clock ticks.
    exclude_median_columns: list
        List of column names for which median should not be computed.
    columns_to_resample: list
        Columns for which first, max, min, last, and median need to be computed.

    Returns:
    resampled_df: pd.DataFrame
        The resampled dataframe.
    """

    # Ensure no NaN values in the event clock column
    df[event_clock_column].fillna(0, inplace=True)

    # Create event clock
    df['clockEvents'] = (df[event_clock_column].cumsum() // event_clock_threshold).astype(int)

    # Create OHLC dictionary
    ohlc_dict = {arg: ['first', 'max', 'min', 'last'] for arg in columns_to_resample}

    # Add Open operation using lambda function
    ohlc_dict.update({col: ['first', 'max', 'min', 'last'] + [lambda x: x.iloc[0]] for col in columns_to_resample})

    # Create median dictionary only for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Remove excluded columns from the list for median calculation
    median_cols = [col for col in numeric_cols if col not in exclude_median_columns]
    median_dict = {col: 'median' for col in median_cols}
    # Add event clock column to dictionary
    event_clock_dict = {event_clock_column: 'sum'}
    # Merge dictionaries
    agg_dict = {**ohlc_dict, **median_dict, **event_clock_dict}
    # Resample the dataframe
    resampled_df = df.groupby('clockEvents').agg(agg_dict)
    # Compute Open, High, Low, Close values for each column in args
    for arg in columns_to_resample:
        resampled_df[arg + '_open'] = df.groupby('clockEvents')[arg].first()
        resampled_df[arg + '_high'] = df.groupby('clockEvents')[arg].max()
        resampled_df[arg + '_low'] = df.groupby('clockEvents')[arg].min()
        resampled_df[arg + '_close'] = df.groupby('clockEvents')[arg].last()
    # Add Garman-Klass volatility calculation
    resampled_df.columns = ['_'.join(map(str, col)).rstrip("_") for col in resampled_df.columns]
    # Reset the index
    resampled_df.reset_index(drop=True, inplace=True)
    return resampled_df


def process_file(params: dict):
    """
    Read a dataframe from a pickle file, resample it using event clock and save it to another pickle file.

    Parameters:
    params: dict
        A dictionary containing the following keys:
            'input_file': str - The path to the input pickle file.
            'output_folder': str - The path to the output folder.
            'event_clock_column': str - The column name in df to be used as event clock.
            'event_clock_threshold': int - The threshold of cumulative number of events to define clock ticks.
            'columns_to_resample': list - Columns for which first, max, min, last, and median need to be computed.
            'exclude_median_columns': list - List of column names for which median should not be computed.
    """
    input_file = params['input_file']
    output_folder = params['output_folder']
    event_clock_column = params['event_clock_column']
    event_clock_threshold = params['event_clock_threshold']
    columns_to_resample = params['columns_to_resample']
    exclude_median_columns = params['exclude_median_columns']

    df = pd.read_pickle(input_file)
    df = df.fillna(method='ffill')
    resampled_df = resample_event_clock_two(df, event_clock_column, event_clock_threshold, exclude_median_columns,
                                            columns_to_resample)
    # resampled_df = resampled_df.reset_index(drop=True, inplace=True)

    resampled_df = resampled_df.fillna(method='ffill')
    calculator = VolatilityCalculator(rolling_window=3)
    # # Apply volatility calculations on the DataFrame
    resampled_df = calculator.garman_klass_parkinson_vols(resampled_df)
    resampled_df = calculator.calculate_midquote_returns(resampled_df)
    resampled_df = calculator.compute_spreads(resampled_df)
    resampled_df = resampled_df.fillna(method='ffill')
    resampled_df['log_pct_changes'] = (
        resampled_df.MicroPrice_median.pct_change().replace([np.inf, -np.inf], 0).apply(np.log).fillna(
            method='ffill')).replace([np.inf, -np.inf], 0)
    resampled_df['pct_changes'] = (
        resampled_df.MicroPrice_median.pct_change().replace([np.inf, -np.inf], 0))
    resampled_df['simple_vol'] = np.sqrt(resampled_df.pct_changes.rolling(2).std())
    # print(resampled_df.head(10)) #DEBUG
    # print(type(resampled_df))
    output_filename = os.path.join(output_folder, os.path.basename(input_file))
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    resampled_df.to_pickle(output_filename)
    print('stored results:', output_filename)


cleanLOBFolder = '/media/ak/Data/CleanLOBData'
interestRateFuturesFolder = '/media/ak/Data/InterestRateFuturesData'
reconstrLOBFolder = os.path.join(interestRateFuturesFolder, 'ReconstructedLOB')
eventClocksFiles = os.path.join(interestRateFuturesFolder, 'EventClocksFiles')
bars = ['tick', 'volume', 'dollar']
symbols = ['RX1', 'DU1']

if __name__ == "__main__":
    # Load parameter sets from JSON file
    config_filepath = '/media/ak/Data/InterestRateFuturesData/EventClocksFiles/configmany.json'
    with open(config_filepath, 'r') as f:
        parameter_sets = json.load(f)

    # Create a pool of workers
    with Pool() as p:
        # Apply process_file to each set of parameters
        p.map(process_file, parameter_sets)
