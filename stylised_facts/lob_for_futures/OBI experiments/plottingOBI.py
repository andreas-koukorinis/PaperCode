import numpy as np
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_acf_and_histogram(df, lags=20, symbol='Symbol', figuresLocation='/path/to/save/figures'):
    """
    Plots the ACF and a histogram distribution for the median of a pandas DataFrame, side by side.

    Parameters:
    df (pandas.DataFrame): DataFrame with numerical data.
    lags (int): Number of lags to include in the ACF plot.
    symbol (str): Symbol name for the title.
    figuresLocation (str): Path to save the figure.
    """
    # Calculate the median of the DataFrame
    median_series = df.median()

    # Calculate ACF and the confidence interval
    acf, confint = sm.tsa.acf(median_series, nlags=lags, alpha=0.05, fft=True)[:2]

    # Calculate standard deviation
    std_dev = (confint - acf[:, None]).mean(axis=1)

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # ACF plot on the first subplot
    ax1.plot(acf, marker='o', linestyle='--', color='blue', label='ACF')
    ax1.fill_between(range(len(acf)), acf - std_dev, acf + std_dev, color='blue', alpha=0.2)
    ax1.set_xlabel("Time Lag", fontsize=15)
    ax1.set_ylabel("Auto-Correlation", fontsize=15)
    ax1.set_title(f"ACF for {symbol}", fontsize=15)

    ax1.tick_params(axis='both', which='major', labelsize=22)  # Set font size for ticks here

    ax1.grid(True)

    # Histogram with KDE plot on the second subplot
    sns.histplot(median_series, ax=ax2, color='blue', alpha =0.2)
    sns.kdeplot(median_series, ax=ax2, color='red', linestyle='dotted',linewidth=7)
    ax2.set_xlabel("Values", fontsize=15)
    ax2.set_ylabel("Frequency", fontsize=25)
    ax2.set_title(f"Distribution of Median Values", fontsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=12)  # Set font size for ticks here
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    title = f'{symbol}_ACF_and_Distribution.png'
    fig_location = os.path.join(figuresLocation, title)
    plt.savefig(fig_location)
    plt.show()
if __name__ == "__main__":
    reconLOBs = '/media/ak/Data1/InterestRateFuturesData/ReconstructedLOB'
    figuresLocation = '/home/ak/Documents/Research/Papers/StylisedFactsPaper/figures'
    symbols = sorted(os.listdir(reconLOBs))
    symbolFolders = [os.path.join(reconLOBs, symbol) for symbol in symbols]
    directory = 'OrderBookImbalance'
    path = os.path.join(reconLOBs, directory)
    pkl_files = [f for f in os.listdir(path) if str('.pkl') in f]
    for pkl_file in pkl_files:
        symbol_ =(pkl_file.split("_")[0])
        symbol_path = os.path.join(path, pkl_file)
        symbol_obi_df = pd.read_pickle(symbol_path )
        plot_acf_and_histogram(symbol_obi_df, lags=20, symbol=symbol_, figuresLocation=figuresLocation)