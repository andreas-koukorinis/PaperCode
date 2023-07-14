import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def create_median_df(symbol):
    """
    This function reads all the pickle files in a symbol's output folder, concatenates them into one DataFrame,
    calculates the median for each column, and then saves this resulting DataFrame into a pickle file.

    :param symbol: string, symbol of the interest rate future
    """
    try:
        # Define the input and output directories
        input_folder = os.path.join('/media/ak/Data/InterestRateFuturesData/MFDCCA', symbol)
        output_folder = os.path.join('/media/ak/Data/InterestRateFuturesData/MFDCCA', 'median_df_per_symbol')
        os.makedirs(output_folder, exist_ok=True)

        # Get all the pickle files in the input directory
        files = [file for file in os.listdir(input_folder) if file.endswith('.pkl')]

        # Read and concatenate all the DataFrames
        dfs = [pd.read_pickle(os.path.join(input_folder, file)).assign(file=file) for file in files]
        df_concat = pd.concat(dfs).groupby(level=0).median()

        # Save the median DataFrame to a pickle file
        output_file = os.path.join(output_folder, symbol + '_median_lagged_correl.pkl')
        df_concat.to_pickle(output_file)

        # Print the name of the file being saved
        print(f"Saving median DataFrame to {output_file}")

    except Exception as e:
        print(f"An error occurred while processing symbol {symbol}: {e}")


reconLOBs = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB'
symbols = os.listdir(reconLOBs)

# Create a process pool and submit the tasks
with ProcessPoolExecutor() as executor:
    executor.map(create_median_df, symbols)
