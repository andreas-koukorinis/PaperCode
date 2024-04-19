import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import fathon  # Assuming you have the 'fathon' library installed
import numpy as np
import pandas as pd
from fathon import fathonUtils as fu


def load_data_for_date(date, HiVolPath, LoVolPath):
    # Filter files in tickHi list for the given date
    tickHi_files = [f for f in os.listdir(HiVolPath) if str(date) in f]

    # Filter files in tickLo list for the given date
    tickLo_files = [f for f in os.listdir(LoVolPath) if str(date) in f]

    # Create dataframes to store the data
    tick_hi_df = pd.DataFrame()
    tick_lo_df = pd.DataFrame()

    # Load data from tickHi files into tick_hi_df
    for file in tickHi_files:
        if 'tick_hi_regime' in file:
            file_path = os.path.join(HiVolPath, file)
            data = pd.read_pickle(file_path)
            tick_hi_df = pd.concat([tick_hi_df, data], axis=0)

    # Load data from tickLo files into tick_lo_df
    for file in tickLo_files:
        if 'tick_lo_regime' in file:
            file_path = os.path.join(LoVolPath, file)
            data = pd.read_pickle(file_path)
            tick_lo_df = pd.concat([tick_lo_df, data], axis=0)

    return tick_hi_df, tick_lo_df


def calculate_dcca(date, HiVolPath, LoVolPath):
    """
    Calculate DCCA (Detrended Cross-Correlation Analysis) for a given date.

    Args:
        date (str): The date for which DCCA is to be calculated.
        HiVolPath (str): The path to the directory containing high-volume data files.
        LoVolPath (str): The path to the directory containing low-volume data files.

    Returns:
        dict: Two dictionaries containing DCCA results for high-volume and low-volume data.
    """
    tick_hi_df, tick_lo_df = load_data_for_date(date, HiVolPath, LoVolPath)
    df1 = tick_lo_df
    df2 = tick_hi_df
    a1 = np.asanyarray(df1['price_imbalance'])
    b1 = np.asanyarray(df1['median_traded_volume'])

    a2 = np.asanyarray(df2['price_imbalance'])
    b2 = np.asanyarray(df2['median_traded_volume'])

    pydcca_l = fathon.MFDCCA(a1, b1)
    a1_len = len(a1)
    a2_len = len(a2)

    # Calculate the maximum window size as a fraction of the minimum length of a1 and a2
    max_window_size = min(a1_len, a2_len) // 2  # Adjust the fraction as needed

    # Create winSizes based on the maximum window size and step
    step = 50  # Adjust the step size as needed
    winSizes = fu.linRangeByStep(20, max_window_size, step=step)
    qs = np.arange(-3, 4, 0.1)
    revSeg = True

    # Now, winSizes is linked to the size of the input arrays a1 and a2

    polOrd = 1

    n_l, F_l = pydcca_l.computeFlucVec(winSizes, qs, revSeg=revSeg, polOrd=polOrd)

    H_l, H_intercept_l = pydcca_l.fitFlucVec()
    # n_l, rho_l = pydcca_l.computeRho(winSizes, polOrd=polOrd)
    alpha_l, mfSpect_l = pydcca_l.computeMultifractalSpectrum()
    tau_l = pydcca_l.computeMassExponents()

    pydcca_h = fathon.MFDCCA(a2, b2)

    n_h, F_h = pydcca_h.computeFlucVec(winSizes, qs, revSeg=revSeg, polOrd=polOrd)

    H_h, H_intercept_h = pydcca_h.fitFlucVec()
    # n_h, rho_h = pydcca_h.computeRho(winSizes, polOrd=polOrd)
    tau_h = pydcca_h.computeMassExponents()
    alpha_h, mfSpect_h = pydcca_h.computeMultifractalSpectrum()

    result_dict_hi = defaultdict(dict)
    result_dict_lo = defaultdict(dict)

    result_dict_hi[date]['H'] = H_h
    result_dict_hi[date]['H_intercept'] = H_intercept_h
    result_dict_hi[date]['n'] = n_h
    # result_dict_hi[date]['rho'] = rho_h
    result_dict_hi[date]['F'] = F_h
    result_dict_hi[date]['tau'] = tau_h
    result_dict_hi[date]['alpha'] = alpha_h
    result_dict_hi[date]['mfSpect'] = mfSpect_h

    result_dict_lo[date]['H'] = H_l
    result_dict_lo[date]['H_intercept'] = H_intercept_l
    result_dict_lo[date]['n'] = n_l
    # result_dict_lo[date]['rho'] = rho_l
    result_dict_lo[date]['F'] = F_l
    result_dict_lo[date]['tau'] = tau_l
    result_dict_lo[date]['alpha'] = alpha_l
    result_dict_lo[date]['mfSpect'] = mfSpect_l

    return result_dict_hi, result_dict_lo


## common dates ##
def extract_dates_from_filenames(file_list):
    dates = []
    for file in file_list:
        # Assuming the date is the last 8 characters of the file name
        date_str = file[-12:-4]
        dates.append(date_str)
    return dates


def find_common_dates(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_dates = set1.intersection(set2)
    return list(common_dates)


if __name__ == "__main__":
    HLPath = '/media/ak/T7/August11th2022Experiments/HiLoData/'
    allHILOData = os.listdir('/media/ak/T7/August11th2022Experiments/HiLoData/')
    symbols = ['RX1', 'TY1', 'G_1',
               'XM1', 'US1', 'FB1', 'KE1 ',
               'JB1', 'TU1', 'OE1', 'YM1']  # DU1 is problematic
    symbol = 'KE1'
    bar = 'volume'
    regimeDirectories = [f for f in allHILOData if str(symbol) in f and 'Regime' in f]

    HiVolPath = (os.path.join(HLPath, regimeDirectories[0]))
    LoVolPath = (os.path.join(HLPath, regimeDirectories[1]))

    tickHi = [f for f in os.listdir(HiVolPath) if str(bar) in f]
    tickLo = [f for f in os.listdir(LoVolPath) if str(bar) in f]
    # Extract dates from the file names
    tickHi_dates = extract_dates_from_filenames(tickHi)
    tickLo_dates = extract_dates_from_filenames(tickLo)

    # Find common dates
    common_dates = find_common_dates(tickHi_dates, tickLo_dates)  # list of common dates

    num_processes = 4  # Adjust the number of processes as needed

    with Pool(num_processes) as pool:
        results_hi, results_lo = zip(
            *pool.starmap(calculate_dcca, [(date, HiVolPath, LoVolPath) for date in common_dates]))

    dict_hi = defaultdict(dict)
    dict_lo = defaultdict(dict)

    for date, data in zip(common_dates, results_hi):
        dict_hi[date] = data

    for date, data in zip(common_dates, results_lo):
        dict_lo[date] = data

    # Define file paths to save the dictionaries in the specified directory
    dict_hi_file = os.path.join(HLPath, str(symbol) + '_' + str(bar) + '_mfdcca_dict_hi.pkl')
    dict_lo_file = os.path.join(HLPath, str(symbol) + '_' + str(bar) + '_mfdcca_dict_lo.pkl')

    # Serialize and save dict_hi to a binary file
    with open(dict_hi_file, 'wb') as file:
        pickle.dump(dict_hi, file)

    # Serialize and save dict_lo to a binary file
    with open(dict_lo_file, 'wb') as file:
        pickle.dump(dict_lo, file)
