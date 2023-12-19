import os
import pandas as pd
import pickle
import concurrent.futures
import numpy as np
import warnings
from collections import defaultdict
from fathon import fathonUtils as fu, MFDFA

def process_file(file_path, qs, revSeg, polOrd, output_directory):
    """
    Process a single file for MFDFA and save the results.
    """
    all_obi_values_df = pd.read_pickle(file_path)
    symbol = os.path.basename(file_path).split("_")[0]
    obi_mfdfa_output_dict_ = defaultdict(dict)

    for index in range(len(all_obi_values_df.columns)):
        data = all_obi_values_df.iloc[:, index]
        try:
            data_length = len(data)
            max_window_size = data_length - 1
            winSizes = fu.linRangeByStep(10, min(max_window_size, 2000))
            a = fu.toAggregated(data)

            pymfdfa = MFDFA(a)

            n, F = pymfdfa.computeFlucVec(winSizes, qs, revSeg=revSeg, polOrd=polOrd)
            if np.any(F <= 0):
                print(f"Warning: Non-positive fluctuation values encountered in {symbol}, index {index}.")
                continue
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    list_H, list_H_intercept = pymfdfa.fitFlucVec()
                alpha, mfSpect = pymfdfa.computeMultifractalSpectrum()
                obi_mfdfa_output_dict_[index] = {
                    'orderBookImbalance': data,
                    'n': n,
                    'F': F,
                    'list_H': list_H,
                    'list_H_intercept': list_H_intercept,
                    'tau': pymfdfa.computeMassExponents(),
                    'alpha': alpha,
                    'mfSpect': mfSpect,
                    'qs': qs
                }
            output_file_name = f"{symbol}_obi_mfdfa.pkl"
        except IndexError as e:
            print(f"Error processing file {file_path}: {e}")
            return f"Error in {file_path}"

    output_loc = os.path.join(output_directory, output_file_name)  # Corrected output location
    with open(output_loc, 'wb') as file:
        pickle.dump(obi_mfdfa_output_dict_, file, protocol=pickle.HIGHEST_PROTOCOL)
    return output_file_name


if __name__ == "__main__":
    reconLOBs = '/media/ak/Data1/InterestRateFuturesData/ReconstructedLOB'
    directory = 'OrderBookImbalance'
    path = os.path.join(reconLOBs, directory)
    output_path = '/media/ak/Data1/InterestRateFuturesData/ReconstructedLOB/OrderBookImbalance/OBIMFDFA'  # Corrected output path
    qs = np.arange(-3, 4, 0.1)
    revSeg = True
    polOrd = 1

    obiFiles = os.listdir(path)
    obiValueFiles = [f for f in obiFiles if '_calendar.pkl' in f]
    file_paths = [os.path.join(path, f) for f in obiValueFiles if f in ['RX1_calendar.pkl', 'TY1_calendar.pkl']]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path, qs, revSeg, polOrd, output_path) for file_path in file_paths]
        for future in concurrent.futures.as_completed(futures):
            output_file = future.result()
            print(f"Completed: {output_file}")
