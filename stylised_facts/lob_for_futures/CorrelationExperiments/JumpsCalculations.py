import time
import numpy as np
from scipy.stats import norm
import pandas as pd
import os
import pickle
from multiprocessing import Pool


def makeReturns(pData):
    return pData.pct_change().fillna(0)


def ait_sahalia_jacod_test(price_series, delta):
    """
    Function to perform the Ait-Sahalia and Jacod (2009) test for jumps.

    Parameters:
    price_series : array-like
        The price series to test.
    delta : float
        The bandwidth parameter.

    Returns:
    df_result : DataFrame
        Contains the z_value, critical_value, and p-value.
    """
    # Create DataFrame with results
    df_result = pd.DataFrame(columns=['z_value', 'critical_value', 'p-value'])

    # Calculate log returns
    log_returns = np.log(price_series[1:]) - np.log(price_series[:-1])

    # Calculate bipower variation
    bipower_variation = (np.pi / 2) * pd.Series(log_returns).abs().rolling(window=2).sum().dropna()

    # Calculate the test statistic (Z-value here)
    z_value = (log_returns ** 2 - delta * bipower_variation) / np.sqrt(
        (delta ** 2) * (2 * np.pi - delta) * (bipower_variation / delta))

    # Calculate the critical value for a 95% confidence level
    critical_value = norm.ppf(0.975)  # Two-tailed test, so we look at 0.975 percentile

    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(np.abs(z_value)))

    # Update DataFrame with calculated values
    df_result['z_value'] = z_value
    df_result['critical_value'] = critical_value
    df_result['p-value'] = p_value

    return df_result


def store_jumps_test(df, jumpsDirectory, FileName):
    pickle_out_returns = os.path.join(jumpsDirectory, str(FileName) + ".pkl")
    pickle.dump(df, open(pickle_out_returns, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def process_symbol(symbol):
    writeSymbolFolder = os.path.join(jumpsDirectory, symbol)
    readSymbolFolder = os.path.join(reconLOBs, symbol)

    files = os.listdir(readSymbolFolder)
    for file in files:
        filePath = os.path.join(readSymbolFolder, file)
        dfLOB = pd.read_pickle(filePath)
        microPrice = dfLOB.MicroPrice
        # s = makeReturns(microPrice)
        df_result = ait_sahalia_jacod_test(microPrice, delta=0.1)
        df_result.to_pickle(os.path.join(writeSymbolFolder, file), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    jumpsDirectory = '/media/ak/Data/InterestRateFuturesData/Jumps'
    reconLOBs = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB'
    symbolsL = ['DU1', 'FB1', 'FV1', 'JB1', 'KE1', 'OE1', 'RX1', 'TY1',
                'US1', 'UST2y', 'XM1', 'YM1', 'UST5y', 'UST10y']

    # Create a pool of worker processes
    pool = Pool()

    # Process symbols in parallel
    pool.map(process_symbol, symbolsL)

    # Close the pool to indicate that no more tasks will be added
    pool.close()

    # Wait for all worker processes to finish
    pool.join()

    # jumpsDirectory = '/media/ak/Data/InterestRateFuturesData/Jumps'
    # reconLOBs = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB'
    # symbolsL = ['DU1', 'FB1', 'FV1', 'JB1', 'KE1', 'OE1', 'RX1', 'TY1',
    #             'US1', 'UST2y', 'XM1', 'YM1', 'UST5y', 'UST10y']
    # for symbol in symbolsL:
    #     writeSymbolFolder = os.path.join('/media/ak/Data/InterestRateFuturesData/Jumps/', symbol)
    #     readSymbolFolder = os.path.join(reconLOBs, symbol)
    #
    #     files = os.listdir(readSymbolFolder)
    #     for file in files:
    #         filePath = os.path.join(readSymbolFolder, file)
    #         dfLOB = pd.read_pickle(filePath)
    #         microPrice = dfLOB.MicroPrice
    #         s = makeReturns(microPrice)
    #         df_result = ait_sahalia_jacod_test(microPrice)
    #         df_result.to_pickle(os.path.join(writeSymbolFolder, file), protocol=pickle.HIGHEST)
