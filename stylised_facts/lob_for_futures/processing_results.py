from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import numpy as np
import os
import pandas as pd


def process_entry(args):
    key, value = args
    result = {}

    if value['results'] != {}:
        result[key] = {}
        result[key]['results'] = value['results']
        result[key]['start_point'] = value['start_point']
        result[key]['end_point'] = value['end_point']
        result[key]['window'] = value['window']

    result[key] = compute_medians(result.get(key, {}))

    return key, result



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


# def main():
#     quadAlpha = '/media/ak/My Passport/QuadAlpha'
#     symbol = 'XM1'
#     bar_choice = 'tick'
#
#     files = [f for f in os.listdir(os.path.join(quadAlpha)) if str(symbol) in f]
#     varFile = [f for f in files if str(bar_choice) in f][0]
#     varFileLoc = os.path.join(os.path.join(quadAlpha), varFile)
#
#     dictionary = pd.read_pickle(varFileLoc)
#     new_dictionary = defaultdict(dict)
#
#     list_of_new_dict_keys = list(dictionary.keys())
#
#     with ProcessPoolExecutor() as executor:
#         results = list(executor.map(process_entry, dictionary.items()))
#
#     for key, value in results:
#         new_dictionary[key].update(value)
#
#     print(new_dictionary)
import pickle

def iterate_and_remove_empty_entries(dictionary):
    new_dictionary = {}
    for key, value in dictionary.items():
        if value and not isinstance(value, dict) or not value:
            new_dictionary[key] = value
    return new_dictionary
def remove_empty_results(dictionary):
    return {k: v for k, v in dictionary.items() if v.get('results')}


def main():
    quadAlpha = '/media/ak/My Passport/QuadAlpha'
    symbol = 'G_1'
    bar_choice = 'dollar'
    target_dir = quadAlpha  # Set your target directory here

    files = [f for f in os.listdir(os.path.join(quadAlpha)) if str(symbol) in f]
    varFile = [f for f in files if str(bar_choice) in f][0]
    varFileLoc = os.path.join(os.path.join(quadAlpha), varFile)

    dictionary = pd.read_pickle(varFileLoc)
    new_dictionary = defaultdict(dict)

    list_of_new_dict_keys = list(dictionary.keys())

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_entry, dictionary.items()))

    for key, value in results:
        new_dictionary[key].update(value)

    # Save the new_dictionary as a pickle file
    new_dict_name = "_".join((symbol, bar_choice,'quad_mmd_consold_results.pkl'))
    # Remove empty entries from new_dictionary
    final_dictionary = iterate_and_remove_empty_entries(new_dictionary)

    with open(os.path.join(target_dir,new_dict_name), 'wb') as f:
        pickle.dump(final_dictionary, f)
    print(final_dictionary)


if __name__ == "__main__":
    main()
