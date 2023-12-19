import numpy as np
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

def process_column(allSymbolsPath, symbolidx, fileidx, columnName):
    """
    Extracts a specific column from a file in a nested directory structure.

    Parameters:
    allSymbolsPath (str): The base path to the directory containing the symbols.
    symbolidx (int): Index to select the symbol directory from the sorted list of directories.
    fileidx (int): Index to select the file from the sorted list of files in the symbol directory.
    columnName (str): The name of the column to be retrieved from the file.

    Returns:
    pandas.Series: The specified column from the file as a pandas Series.

    Raises:
    IndexError: If symbolidx or fileidx is out of range.
    FileNotFoundError: If the file or directory does not exist.
    """

    # Get the list of symbol directories, sorted to ensure consistent ordering
    symbols = sorted(os.listdir(allSymbolsPath))
    if symbolidx >= len(symbols):
        raise IndexError("symbolidx is out of range")

    symbol = symbols[symbolidx]

    # Construct the path to the symbol directory
    symbol_dir_path = os.path.join(allSymbolsPath, symbol)

    # Get the list of files in the symbol directory, sorted to ensure consistent ordering
    files = sorted(os.listdir(symbol_dir_path))
    if fileidx >= len(files):
        raise IndexError("fileidx is out of range")

    file_name = files[fileidx]

    # Construct the full file path
    file_path = os.path.join(symbol_dir_path, file_name)

    # Load the specified file
    try:
        load_file = pd.read_pickle(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Retrieve and return the specified column
    try:
        return load_file[columnName]
    except KeyError:
        raise KeyError(f"Column {columnName} not found in the file")


if __name__ == "__main__":
    reconLOBs = '/media/ak/Data1/InterestRateFuturesData/ReconstructedLOB'
    symbols = sorted(os.listdir(reconLOBs))
    symbolFolders = [os.path.join(reconLOBs, symbol) for symbol in symbols]
    allSymbolsPath = reconLOBs
    symbolidx = 16
    columnName = 'OrderImbalance'

    # Get the list of files in the specified symbol directory
    symbol = sorted(os.listdir(allSymbolsPath))[symbolidx]
    print(symbol)
    symbol_dir_path = os.path.join(allSymbolsPath, symbol)
    files = sorted(os.listdir(symbol_dir_path))

    # Dictionary comprehension to apply the function for each file
    # Dictionary comprehension to apply the function for each file, using the index as the key
    column_data = {fileidx: process_column(allSymbolsPath, symbolidx, fileidx, columnName).reset_index(drop=True)
                   for fileidx in range(len(files))}

    column_df = pd.DataFrame.from_dict(column_data).dropna()

    print(column_df)
    # Path where you want to save the file
    save_path = '/media/ak/Data1/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance'

    # Ensure the directory exists, if not, create it
    os.makedirs(save_path, exist_ok=True)
    current_clock = 'calendar'

    # Construct the filename with symbol and time
    filename = f"{symbol}_{str(current_clock)}.pkl"  # Change to .pkl if you want to save as pickle

    # Full path for the file
    full_file_path = os.path.join(save_path, filename)

    # Save the DataFrame
    # As CSV
    # column_df.to_csv(full_file_path)

    # Or as a pickle file (uncomment below if you prefer pickle)
    column_df.to_pickle(full_file_path)

    print(f"File saved as {full_file_path}")