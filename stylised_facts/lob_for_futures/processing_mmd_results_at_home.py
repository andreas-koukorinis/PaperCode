import numpy as np
import os
import pandas as pd
import pickle
from collections import defaultdict

quad_mmd_output_files = '/media/ak/T7/August11th2022Experiments/QuadMMDOutputFiles'


def pickle_dump_obj_to_filename(destinationPath, symbol, fileName, obj):
    pickle_out_filename = os.path.join(destinationPath, "_".join((symbol, fileName)))
    with open(pickle_out_filename, 'wb') as pickle_out:
        pickle.dump(obj, pickle_out)
    print('saved', pickle_out_filename)

def iterate_and_remove_empty_entries(some_dict):
    cleaned_dict = {}
    for key, value in some_dict.items():
        if isinstance(value, dict):
            cleaned_value = iterate_and_remove_empty_entries(value)  # recursion
            if cleaned_value:  # include this key-value pair only if cleaned_value is not None or not empty
                cleaned_dict[key] = cleaned_value
        elif isinstance(value, np.ndarray):
            if value.any():  # include this key-value pair only if any value in the array is not None or not empty
                cleaned_dict[key] = value
        elif value:  # for non-array values, include this key-value pair only if value is not None or not empty
            cleaned_dict[key] = value
    return cleaned_dict


def process_dictionary_into_dataframe(clean_dictionary):
    data = []
    columns = ['perm_mmds1', 'chi2_weights', 'sim_mmds', 'sig2', 'Kxy', 'mean', 'var',
               'Kxx', 'Kyy', 'mean_gram', 'var_gram', 'med', 'list_gwidth', 'besti',
               'powers', 'best_ker', 'XZ_test', 'YZ_test']

    for key, val in clean_dictionary.items():
        row = {column: np.nan for column in columns}
        index_values = tuple(key)  # start_point, end_point, shift, window
        results = val.get('results', {})
        for result_key, result_val in results.items():
            if result_key in columns:
                if isinstance(result_val, list):
                    # compute the median of the list
                    row[result_key] = np.median(result_val)
                else:
                    row[result_key] = result_val
        row.update({'index_values': index_values})
        data.append(row)

    df = pd.DataFrame(data).set_index('index_values')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = ['start_point', 'end_point', 'shift', 'window']
    return df

if __name__ == "__main__":
    symbol = 'TY1'
    var = 'tau'
    quadAlpha = os.path.join(quad_mmd_output_files, var)
    files = [f for f in os.listdir(quadAlpha) if str(symbol) in f]
    bar_choice = 'tick'
    varFile = [f for f in files if str(bar_choice) in f][0]
    varFileLoc = os.path.join(quadAlpha, varFile)
    print(varFileLoc)
    pickled_dict = pd.read_pickle(varFileLoc)
    output_directory = '/media/ak/T7/August11th2022Experiments/QuadMMDOutputFiles/processedTauResults/'
    # clean_directory = ((iterate_and_remove_empty_entries(pickled_dict)))
    clean_directory_keys =list(pickled_dict.keys())
    for key in clean_directory_keys:
        value = pickled_dict[key]  # Extract value
        if not value['results']:  # Check if 'results' sub-dict is empty
            print(f'Empty dict at key {key}')
        else:
            print(key)  # If not, print the dictionary
            filename = f"{bar_choice}_{int(''.join(map(str, key)))}_{var}results.pickle"
            pickle_dump_obj_to_filename(output_directory, symbol, filename, value)
            #print(value)

