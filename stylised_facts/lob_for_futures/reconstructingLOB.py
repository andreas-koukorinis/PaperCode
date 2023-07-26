import os
import pandas as pd
from functools import partial
import multiprocessing as mp
import numpy as np
import concurrent.futures
import pickle


class LOBConstructor:
    """
    Creating the LOBConstructor class.
    This class has methods to locate the trade and quote files, process these files, and merge the processed data.
    The __init__ method initializes the instance with a path to the data
    and a symbol (the financial instrument being tracked).
    The data is assumed to be seperated by Quotes and Trades. But nothing beyond that.
    """

    def __init__(self, data_path, symbol):
        self.data_path = data_path
        self.symbol = symbol
        self.file_dict = None
        self.trades_df = None
        self.quotes_df = None

    def locate_files(self):
        symbol_trades_folder = os.path.join(self.data_path, self.symbol, self.symbol + '_Comdty_trades')
        symbol_quotes_folder = os.path.join(self.data_path, self.symbol, self.symbol + '_Comdty_quotes')

        # Retrieve the files in the trades and quotes folders
        trades_files = os.listdir(symbol_trades_folder)
        quotes_files = os.listdir(symbol_quotes_folder)

        # Extract the common dates from the files
        common_dates = set([os.path.splitext(file)[0] for file in trades_files]) & set(
            [os.path.splitext(file)[0] for file in quotes_files])

        # Create a dictionary with common dates as keys and corresponding date.csv files as values
        file_dict = {}
        for date in common_dates:
            trades_file = os.path.join(symbol_trades_folder, f'{date}.csv')
            quotes_file = os.path.join(symbol_quotes_folder, f'{date}.csv')
            file_dict[date] = (trades_file, quotes_file)

        self.file_dict = file_dict
        return self.file_dict

    def process_trades_file(self, trades_file):
        trades_file, _ = trades_file  # Unpack the trades_file from the tuple
        trades_df = pd.read_csv(trades_file)

        # Sort the trades by timestamp in ascending order
        trades_df['time'] = pd.to_datetime(trades_df['time'])
        # Create the 'date' column

        trades_df.sort_values(by='time', inplace=True)  # sorting in chronological order

        # Calculate inter-arrival time and estimated arrival time
        trades_df['inter_arrival_time'] = trades_df['time'].diff()
        trades_df['estimated_arrival_time'] = trades_df['time'].iloc[0] + trades_df['inter_arrival_time'].cumsum()

        # Perform the groupby operation and aggregate the data
        trades_df = trades_df.groupby(['time', 'estimated_arrival_time', 'value']).agg(
            {'size': 'sum', 'type': 'count'}).reset_index()
        # grouping by time, estimated arrival time and value so any partial executions are aggregated
        trades_df.rename(columns={'size': 'TradedVolume', 'type': 'NoOfTrades'}, inplace=True)
        # Calculate the total time duration covered by the trades
        total_time_duration = trades_df['estimated_arrival_time'].max() - trades_df['estimated_arrival_time'].min()
        # this is total time duration covered by the trades on that day.
        # used this for arrival time and rate

        # Calculate the arrival rate for each trade and add it as a column
        trades_df['arrival_rate'] = trades_df['NoOfTrades'] / total_time_duration.total_seconds()
        trades_df['AverageVolumePerTrade'] = trades_df['TradedVolume'] / trades_df['NoOfTrades']
        # calculating the volume traded per trade

        trades_df['date'] = trades_df['time'].dt.date
        trades_df['CCYTradedVolume'] = trades_df['TradedVolume'] * trades_df['value']
        # computing the CCY value of the traded volume!
        # Assuming 'TradedVolume' refers to the traded volume and then the value is the traded price

        return trades_df

    def read_trades_files(self):
        with mp.Pool(mp.cpu_count()) as pool:
            process_trades_file_partial = partial(self.process_trades_file)
            trades_dfs = pool.map(process_trades_file_partial, self.locate_files().values())

        # Extract the DataFrame from each tuple in trades_dfs
        trades_dfs = [df for df in trades_dfs]

        self.trades_df = pd.concat(trades_dfs, ignore_index=True)
        self.trades_df = self.trades_df.rename(columns={'value': 'TradePrice'})
        return self.trades_df

    def process_one_quotes_file(self, one_quotes_file):
        quotes_df = pd.read_csv(one_quotes_file)
        quotes_df['time'] = pd.to_datetime(quotes_df['time'])
        # Sort the quotes by timestamp in ascending order
        quotes_df.sort_values(by='time', inplace=True)
        # Group quotes by type, value, and time and aggregate the data
        quotes_df = quotes_df.groupby(['type', 'value', 'time']).agg({'size': 'sum'}).reset_index()
        # If 'value' equals 0, replace it with the previous non-zero 'value'
        quotes_df['value'].replace(0, np.nan, inplace=True)
        quotes_df['value'].fillna(method='ffill', inplace=True)
        quotes_df['date'] = quotes_df['time'].dt.date
        quotes_df.fillna(method='ffill', inplace=True)

        return quotes_df

    def process_all_quotes_files(self):
        self.files_dict = self.locate_files()
        dfs_list = []
        # Create a process pool and process each file in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for df in executor.map(self.process_one_quotes_file, [f[1] for f in self.files_dict.values()]):
                dfs_list.append(df)

        all_dfs = pd.concat(dfs_list, ignore_index=True)
        all_dfs.sort_values(by='time', inplace=True)
        all_dfs = all_dfs.assign(BestBid=all_dfs['value'].where(all_dfs['type'] == 'BID', None),
                                 BidSize=all_dfs['size'].where(all_dfs['type'] == 'BID', None),
                                 BestAsk=all_dfs['value'].where(all_dfs['type'] == 'ASK', None),
                                 AskSize=all_dfs['size'].where(all_dfs['type'] == 'ASK', None))
        all_dfs.fillna(method='ffill', inplace=True)

        return all_dfs

    @staticmethod
    def constructLOB(trades_df, quotes_df):
        merged_df = pd.concat([trades_df, quotes_df], axis=0)
        merged_df.sort_values(by='time', inplace=True)
        merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
        # Replacing NaN values in 'Column1' with 'Trade'
        merged_df['type'].fillna('Trade', inplace=True)
        merged_df['CumlTradingVolume'] = merged_df.groupby('date')[
            'TradedVolume'].cumsum()  # Calculating running of actual trading volume
        merged_df['CCYCumlVolume'] = merged_df.groupby('date')['CCYTradedVolume'].cumsum()  # Calculating running sum
        merged_df['CumlNoOfTrades'] = merged_df.groupby('date')['NoOfTrades'].cumsum()  # Calculating running sum
        merged_df['CumlTradedVolume'] = merged_df.groupby('date')['TradedVolume'].cumsum()  # Calculating running sum
        merged_df['RunningNoOfTrades'] = merged_df['NoOfTrades'].fillna(
            method='ffill')  # Filling NaNs in 'NoOfTrades' column --> these must follow the calculations
        merged_df['RunningCumlVolume'] = merged_df['CCYCumlVolume'].fillna(
            method='ffill')  # Filling NaNs in 'NoOfTrades' column --> these must follow the calculations
        merged_df['RunningCumlNoOfTrades'] = merged_df['CumlNoOfTrades'].fillna(
            method='ffill')  # Filling NaNs in 'NoOfTrades' column --> these must follow the calculations

        return merged_df


def pickle_dfs_by_date(df, directory):
    """
    This function saves a DataFrame as a pickled file.
    The filename is based on the date contained in the DataFrame and the provided directory path.
    :param df: this is the DataFrame that will be saved. it is for each unique date in the merged DataFrame.
    :param directory: this is the symbol directory where the pickled file will be saved.
    :return: the fileName so I can create a hash file for each date in the merged DataFrame.
    """
    fileName = os.path.join(directory, f"{list(df.date)[0].strftime('%Y%m%d')}.pkl")
    with open(fileName, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return fileName


def microstructureFeatures(df):
    """
    This function computes microstructure features for a dataframe of tick data.

    df : pandas DataFrame
        The input dataframe. Should include 'BestBid', 'AskSize', 'BestAsk',
        'BidSize', 'TradedVolume' columns.

    Returns
    -------
    df : pandas DataFrame
        The same dataframe that was input, but with new columns added:
        'MicroPrice', 'BASpread', 'QuoteSize', 'MidPrice', 'OrderImbalance',
        'ExpandingMeanRelativeTickVolume', and 'MeanRelativeTickVolume'.
    """

    # MicroPrice is a weighted average of the best bid and best ask prices, where the weights are the sizes of the
    # quotes.
    df = df.copy()  # Make a copy of the DataFrame slice
    df['MicroPrice'] = (df['BestBid'] * df['AskSize'] + df['BestAsk'] * df['BidSize']) / (df['BidSize'] + df['AskSize'])

    # BASpread (Bid-Ask Spread) is the difference between the best ask price and the best bid price.
    df['BASpread'] = df['BestAsk'] - df['BestBid']

    # QuoteSize is the total size of both the best bid and best ask quotes.
    df['QuoteSize'] = df['BidSize'] + df['AskSize']

    # MidPrice is the average of the best bid and best ask prices.
    df['MidPrice'] = (df['BestBid'] + df['BestAsk']) / 2

    # OrderImbalance is the difference in size of the best bid and best ask quotes, normalized by the total quote size.
    df['OrderImbalance'] = (df['BidSize'] - df['AskSize']) / (df['BidSize'] + df['AskSize'])

    # ExpandingMeanRelativeTickVolume is the traded volume divided by the expanding mean of the traded volume. The
    # expanding mean is the average of all prior values, so this measures how the current traded volume compares to
    # the average up to this point.
    df['ExpandingMeanRelativeTickVolume'] = df['TradedVolume'].fillna(0) / df['TradedVolume'].fillna(
        0).expanding().mean()

    # MeanRelativeTickVolume is the traded volume divided by the overall mean traded volume.
    # This measures how the current traded volume compares to the overall average volume.
    df['MeanRelativeTickVolume'] = df['TradedVolume'].fillna(0) / df['TradedVolume'].fillna(0).mean()

    # Fill missing values using forward fill (ffill)
    df.fillna(method='ffill', inplace=True)

    return df


def process_symbol(symbol, data_path_, base_directory_,
                   reconstructed_lob_directory_):
    symbolDirectory_ = os.path.join(reconstructed_lob_directory_, str(symbol))

    # Create an instance of LOBConstructor
    lob = LOBConstructor(data_path_, symbol)
    # Read and process the trades files
    trades_df = lob.read_trades_files()

    # Read and process the quotes files
    quotes_df = lob.process_all_quotes_files()
    merged_df = lob.constructLOB(trades_df, quotes_df)
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    stored_files = dict()
    for date in merged_df['date'].unique():
        df_sub = merged_df[merged_df['date'] == date]
        df_sub = microstructureFeatures(df_sub)
        stored_files[date.strftime('%Y%m%d')] = pickle_dfs_by_date(df_sub, symbolDirectory_)
    fileName = os.path.join(base_directory_, str(symbol) + "_cleanLOBLocs.pkl")
    with open(fileName, 'wb') as handle:
        pickle.dump(stored_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    merged_df_FileName = os.path.join(base_directory_, str(symbol) + '_AllDataFrames.pkl')
    with open(merged_df_FileName, 'wb') as handle:
        pickle.dump(merged_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # storing all the dataframes in a pickle file as I will need them for exp 2,3
    return fileName


if __name__ == "__main__":
    data_path = '/media/ak/Data/InterestRateFuturesData/QuotesTrades/'
    base_directory = '/media/ak/Data/InterestRateFuturesData'
    reconstructed_lob_directory = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB/'
    symbols = ['UST2y']  # This is the symbols list
    '''
    This code creates a list of symbols that you want to process. 
    Then it uses a ProcessPoolExecutor to process each symbol in a separate process. 
    This will speed up the execution time on a multi-core machine.
    The function process_symbol is a refactoring of your original code that processes a single symbol. 
    It takes a symbol as input and processes it as before.

    The ProcessPoolExecutor is used in a context manager (with statement), 
    so the processes will be automatically cleaned up when they're done.
    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_symbol = {
            executor.submit(process_symbol, symbol, data_path, base_directory, reconstructed_lob_directory): symbol for
            symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                print(f"Symbol {symbol} processed successfully. Stored in {data}")
            except Exception as exc:
                print(f"Symbol {symbol} generated an exception: {exc}")
