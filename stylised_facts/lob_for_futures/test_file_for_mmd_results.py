import numpy as np
import os
import pandas as pd
import ast
from collections import defaultdict
import json
import re
import yaml
import pickle
#LinearMMDInputFiles = '/media/ak/T7/August11th2022Experiments/LinearMMDInputFiles/'
quad_mmd_output_files = '/media/ak/T7/August11th2022Experiments/QuadMMDOutputFiles'
quadAlpha = quad_mmd_output_files
#'/media/ak/My Passport/QuadAlpha' #- work laptop
#os.listdir(quad_mmd_output_files)
def pickle_dump_obj_to_filename(destinationPath, fileName, obj):
    pickle_out_filename = os.path.join(destinationPath, fileName)
    with open(pickle_out_filename, 'wb') as pickle_out:
        pickle.dump(obj, pickle_out)
    print('Saved', pickle_out_filename)

# def pickle_dump_obj_to_filename(destinationPath, symbol, fileName, obj):
#     """
#     Using this to dump results for a list to file
#     :param destinationPath: path where the pickle should be dumped
#     :param symbol: Symbol that should accompany the file
#     :param fileName: the specific filename you want to use, like OOSResults.pkl
#     :param obj: the object you want to pickle
#     :return: dumps an obj to file
#     """
#     pickle_out_filename = os.path.join(destinationPath, "_".join((symbol, fileName)))
#     pickle_out = open(pickle_out_filename, 'wb')
#     pickle.dump(obj, pickle_out)
#     pickle_out.close()
#     print('saved', pickle_out_filename)

def iterate_and_remove_empty_entries(dictionary):
    """Iterates through a dictionary and removes all the entries for which the
    'results': defaultdict(dict, {})' condition holds and the dictionary is empty.
  Args:
    dictionary: The dictionary to iterate through.
  Returns:
    The new dictionary without the removed entries.
  """
    return {key: value for key, value in dictionary.items() if not (isinstance(value, defaultdict) and not value)}

def compute_medians(data, prefix=''):
    result = {}
    for k, v in data.items():
        key = f'{prefix}{k}' if prefix else k

        if isinstance(v, dict):
            result.update(compute_medians(v, prefix=f'{key}_'))
        elif isinstance(v, np.ndarray):
            try:
                v = v.astype(float)
                result[key] = np.median(v)
            except ValueError:
                print("Array could not be converted to a numeric format.")
                result[key] = v
        elif isinstance(v, (float, np.float64)):
            result[key] = v
        else:
            result[key] = v

    return result

def remove_prefix_from_index(df, prefix):
    df.index = df.index.to_series().str.replace(prefix, '')
    return df

if __name__ == "__main__":
    symbol = 'XM1'
    var = 'alpha'
    quadAlpha = os.path.join(quad_mmd_output_files, var) #quadAlpha
    files = [f for f in os.listdir(os.path.join(quadAlpha)) if str(symbol) in f]
    bar_choice = 'tick'
    varFile = [f for f in files if str(bar_choice) in f][0]
    varFileLoc = os.path.join(os.path.join(quadAlpha), varFile)
    print(varFileLoc)
    pickled_dict = pd.read_pickle(varFileLoc)
    clean_dictionary = (iterate_and_remove_empty_entries(pickled_dict))
    clean_dictionary_keys = list(clean_dictionary.keys())
    for key in clean_dictionary_keys:
        df =pd.DataFrame.from_dict(clean_dictionary[key])
        if not df.empty:
            print(key[0])
            keyToNumber = int(''.join(map(str, key)))
            print(keyToNumber)
            prefix_ = 'results_'
            df_clean = (remove_prefix_from_index(df, prefix_+str(key[0])+'_'))
            print(df_clean)
            # fileName = str(keyToNumber)+'_results_DF.pkl'
            # pickle_dump_obj_to_filename(destinationPath=quadAlpha, fileName=fileName, symbol=symbol, obj=df)

    # print(compute_medians(new_dictionary,{}))
