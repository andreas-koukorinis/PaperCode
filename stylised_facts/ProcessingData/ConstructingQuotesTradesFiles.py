import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Define paths and constants
rawdatafolder = '/media/ak/Data/raw bloomberg data/'
irf = '/media/ak/Data/InterestRateFuturesData'
quotesTrades = os.path.join(irf, 'QuotesTrades')
folderLocList = ['/media/ak/Data/raw bloomberg data/TUI']
#     '/media/ak/Data/raw bloomberg data/FV1_Comdty-20181028',
#     '/media/ak/Data/raw bloomberg data/G_1_Comdty-20181028',
#     '/media/ak/Data/raw bloomberg data/JB1_Comdty-20181028',
#     '/media/ak/Data/raw bloomberg data/KE1_Comdty-20181028',
#     '/media/ak/Data/raw bloomberg data/OE1_Comdty-20181028',
#     '/media/ak/Data/raw bloomberg data/RX1_Comdty-20181028'
# ]


def process_folder(folderLocIdx: int):
    """
    Function to process a single folder:
    1. Reads the CSV files in the folder
    2. Splits the data into trades and quotes
    3. Writes trades and quotes to separate CSV files
    """

    # Get the folder location from the list using index
    folderLoc = folderLocList[folderLocIdx]

    # Extract the symbol from the folder location string
    string = folderLoc.split("/")[-1].split("-")[0].split("_")[0]  # basically the symbol

    # Construct strings for quotes and trades file paths
    quotesString = "_".join((string, 'Comdty_quotes'))
    tradesString = "_".join((string, 'Comdty_trades'))

    # Create a path to store the processed data
    mainSymbolToStorePath = os.path.join(quotesTrades, string)

    # Get list of files in the current folder
    files_list = os.listdir(folderLoc)

    # Loop through the first 5 files
    for f in files_list:
        # Extract the file name to be used in the destination paths
        toStoreName = f.split('-')[-1]

        # Create paths to store quotes and trades data
        quotesmainSymbolToStorePath = os.path.join(mainSymbolToStorePath, quotesString, str(toStoreName))
        tradesmainSymbolToStorePath = os.path.join(mainSymbolToStorePath, tradesString, str(toStoreName))

        # Load the data from the current CSV file into a pandas DataFrame
        df = pd.read_csv(os.path.join(folderLoc, f), index_col=0)

        # Split the data into trades and quotes
        mask = df['type'] == 'TRADE'
        trades = df[mask].copy()  # Create a copy to avoid SettingWithCopyWarning
        quotes = df[~mask].copy()  # Negate the mask to get the other rows

        # Replace 'TRADE' with 'Trade' in the 'type' column of the trades DataFrame
        trades['type'].replace({'TRADE': 'Trade'}, inplace=True)

        # Write the trades and quotes DataFrames to separate CSV files
        trades.to_csv(tradesmainSymbolToStorePath, index=False)
        quotes.to_csv(quotesmainSymbolToStorePath, index=False)
        print("saved trades & quotes to " + quotesmainSymbolToStorePath + "")


# Create a pool of worker processes
with ProcessPoolExecutor() as executor:
    # Apply the process_folder function to every folder in parallel
    # The map function applies a function to every item in an iterable, in this case the list of folder indices
    executor.map(process_folder, range(len(folderLocList)))
