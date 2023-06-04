import numpy as np
import os
import pandas as pd
import ast
from collections import defaultdict
import json
import re
import yaml

LinearMMDInputFiles = '/media/ak/T7/August11th2022Experiments/LinearMMDInputFiles/'
quad_mmd_output_files = '/media/ak/T7/August11th2022Experiments/QuadMMDOutputFiles'
# quadAlpha = '/media/ak/My Passport/QuadAlpha' - work laptop
os.listdir(quad_mmd_output_files)


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


if __name__ == "__main__":
    symbol = 'XM1'
    var = 'alpha'
    quadAlpha = os.path.join(quad_mmd_output_files, var)
    files = [f for f in os.listdir(os.path.join(quadAlpha)) if str(symbol) in f]
    bar_choice = 'tick'
    varFile = [f for f in files if str(bar_choice) in f][0]
    varFileLoc = os.path.join(os.path.join(quadAlpha), varFile)
    print(varFileLoc)
    pickled_dict = pd.read_pickle(varFileLoc)
    new_dictionary = (iterate_and_remove_empty_entries(pickled_dict))
    new_dictionary_keys = list(new_dictionary.keys())
    #new_dictionary[(129, 131, 2, 5)]
    print(compute_medians(new_dictionary,{}))
