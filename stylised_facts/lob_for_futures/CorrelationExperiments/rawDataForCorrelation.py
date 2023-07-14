import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def process_file(symbolFolder, files, idx, symbol, max_lag):
    """
    This function processes a given file, calculates percentage changes of given columns, shifts the data by given lags,
    and then saves the resulting DataFrame as a pickle file in the specified output directory.

    :param symbolFolder: string, path to the folder containing the files for the given symbol
    :param files: list, filenames in the symbolFolder
    :param idx: int, index of the file to be processed
    :param symbol: string, symbol of the interest rate future
    :param max_lag: int, maximum number of lags to shift the data by
    """
    # Load DataFrame
    filePath = os.path.join(symbolFolder, files[idx])
    dfLOB = pd.read_pickle(filePath)

    # Calculate percentage changes
    dfCorrel = pd.DataFrame()
    dfCorrel['microPriceReturns'] = dfLOB.MicroPrice.pct_change().fillna(0)
    dfCorrel['averageVolumePerTrade'] = dfLOB.AverageVolumePerTrade.fillna(0)
    dfCorrel['OrderBookImbalance'] = dfLOB.OrderImbalance.fillna(0)
    dfCorrel['MeanRelativeTickVolume'] = dfLOB.MeanRelativeTickVolume.fillna(0)
    dfCorrel['TradedVolume'] = dfLOB.TradedVolume.fillna(0)

    # Shift the data by different lags and concatenate
    for lag in range(1, max_lag + 1):
        shifted = dfCorrel.shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in shifted.columns]
        dfCorrel = pd.concat([dfCorrel, shifted], axis=1)

    # Save the DataFrame to a pickle file
    output_folder = os.path.join('/media/ak/Data/InterestRateFuturesData/MFDCCA', symbol)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, files[idx])
    dfCorrel.to_pickle(output_file)

    # Print the name of the file being saved
    print(f"Saving {output_file}")

    # Compute the correlation matrix of dfCorrel
    dfCorr = dfCorrel.corr()

    # Save the correlation matrix to a separate pickle file in a different directory
    mainCorreldir = os.path.join('/media/ak/Data/InterestRateFuturesData/MFDCCA', 'RawCorrelations', symbol)
    os.makedirs(mainCorreldir, exist_ok=True)
    output_file_corr = os.path.join(mainCorreldir, files[idx])
    dfCorr.to_pickle(output_file_corr)

    # Print the name of the file being saved
    print(f"Saving correlation matrix to {output_file_corr}")


def process_symbol_folder(symbolFolder, symbol, max_lag):
    """
    This function processes all files in the given symbol folder in parallel.

    :param symbolFolder: string, path to the folder containing the files for the given symbol
    :param symbol: string, symbol of the interest rate future
    :param max_lag: int, maximum number of lags to shift the data by
    """
    files = os.listdir(symbolFolder)
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, [symbolFolder] * len(files), [files] * len(files), range(len(files)),
                     [symbol] * len(files), [max_lag] * len(files))


reconLOBs = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB'
symbols = os.listdir(reconLOBs)
symbolFolders = [os.path.join(reconLOBs, symbol) for symbol in symbols]
max_lag = 2  # set maximum number of lags

# Parallelize the processing of symbol folders
with ProcessPoolExecutor() as executor:
    executor.map(process_symbol_folder, symbolFolders, symbols, [max_lag] * len(symbolFolders))
