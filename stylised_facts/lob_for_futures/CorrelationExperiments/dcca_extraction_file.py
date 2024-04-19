import os
import pandas as pd
import numpy as np
import fathon
from fathon import fathonUtils as fu
import pickle
from concurrent.futures import ProcessPoolExecutor


def process_file(symbol, bar, var1, var2, path, filename, polOrd, listName, HLPath):
    """
    Process a single file, compute DCCA statistics, and store the results.

    Args:
        symbol (str): Symbol of the financial instrument.
        bar (str): Type of bar, e.g., 'volume'.
        var1 (str): First variable for DCCA analysis.
        var2 (str): Second variable for DCCA analysis.
        path (str): Path to the data files.
        filename (str): Name of the file to process.
        polOrd (int): Polynomial order for detrending in DCCA.
        listName (str): 'HiVol' or 'LoVol', indicating the volume type.
        HLPath (str): Path where results should be saved.

    Returns:
        dict: A dictionary containing computed results for the file.
    """
    result_dict = {}
    df = pd.read_pickle(os.path.join(path, filename))
    if df.empty:
        return {filename: 'Insufficient data'}

    try:
        a1, b1 = np.asanyarray(df[str(var1)]), np.asanyarray(df[var2])
        pydcca_l = fathon.DCCA(a1, b1)
        date = filename.split('regime')[1].split(".pkl")[0]
        winSizes = fu.linRangeByStep(20, min(len(a1), len(b1)), step=50)
        n, F = pydcca_l.computeFlucVec(winSizes, polOrd=polOrd)
        result = pydcca_l.fitFlucVec()

        if isinstance(result, tuple):
            H, H_intercept = result
            result_dict[date] = {
                'n': n.tolist(),
                'F': F.tolist(),
                'H': H,
                'H_intercept': H_intercept,
                'nrho': None,
                'rho': None
            }

            nrho, rho = pydcca_l.computeRho(winSizes, polOrd=polOrd)
            result_dict[date]['nrho'] = nrho
            result_dict[date]['rho'] = rho.tolist()
        else:
            result_dict[date] = {'error': result}
    except Exception as e:
        result_dict[date] = {'error': str(e)}

    pathToSave = os.path.join(HLPath, "_".join((symbol, bar, var1, var2, listName, 'DCCA.pkl')))
    with open(pathToSave, 'wb') as file:
        pickle.dump(result_dict, file)

    return result_dict
def process_path(symbol, bar, var1, var2, path, polOrd, listName, HLPath):
    """
    Process all files in a given path in parallel.

    Args:
        symbol, bar, var1, var2, polOrd, listName, HLPath: Parameters for the processing function.
        path (str): Path to the files to be processed.
    """
    barList = [f for f in os.listdir(path) if str(bar) in f]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, symbol, bar, var1, var2, path, f, polOrd, listName, HLPath) for f in
                   barList]
        results = [future.result() for future in futures]


def main():
    symbol = 'TY1'
    allHILOData = '/media/ak/T7/August11th2022Experiments/HiLoData/'
    HLPath = '/media/ak/T7/August11th2022Experiments/HiLoData/'
    bar = 'tick'
    var1 = 'GK_vol'
    var2 = 'median_traded_volume'
    polOrd = 1

    regimeDirectories = [f for f in os.listdir(allHILOData) if str(symbol) in f]
    HiVolPath = os.path.join(allHILOData, regimeDirectories[0])
    LoVolPath = os.path.join(allHILOData, regimeDirectories[1])

    # Process HiVol and LoVol paths in parallel
    with ProcessPoolExecutor() as executor:
        executor.submit(process_path, symbol, bar, var1, var2, HiVolPath, polOrd, 'HiVol', HLPath)
        executor.submit(process_path, symbol, bar, var1, var2, LoVolPath, polOrd, 'LoVol', HLPath)


if __name__ == "__main__":
    main()
