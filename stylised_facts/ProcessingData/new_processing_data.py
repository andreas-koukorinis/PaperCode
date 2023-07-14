from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from collections import OrderedDict
import pickle


def list_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def process_subdirectory(subdir, output_dir):
    ordered_dict = OrderedDict()
    files = list_files_in_directory(subdir)
    for file in files:
        try:
            data = pd.read_pickle(file)
            if isinstance(data, dict):
                for key in data.keys():
                    if isinstance(data[key], dict):
                        for sub_key in data[key].keys():
                            if sub_key in ['tick', 'volume', 'calendar', 'dollar']:
                                if key not in ordered_dict:
                                    ordered_dict[key] = OrderedDict()
                                ordered_dict[key][sub_key] = data[key][sub_key]
        except Exception as e:
            print(f"An error occurred while trying to read file {file}: {e}")

    # Save output into a file in the specified directory
    output_file = os.path.join(output_dir, f'{os.path.basename(subdir)}_output.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(ordered_dict, f)


def process_directory(directory, symbol, output_dir):
    subdir = os.path.join(directory, symbol)
    if os.path.isdir(subdir):
        print(f"Processing subdirectory: {subdir}")
        with ProcessPoolExecutor() as executor:
            executor.submit(process_subdirectory, subdir, output_dir)

    # Load results from the file
    output_file = os.path.join(output_dir, f'{symbol}_output.pkl')
    with open(output_file, 'rb') as f:
        result = pickle.load(f)

    return result


if __name__ == '__main__':
    t7 = '/media/ak/T7'
    bars = ['tick', 'volume', 'calendar', 'dollar']
    rawDataFolder = os.path.join(t7, 'FuturesDataSemiProcessed')
    symbol = 'RX1'  # replace with your symbol
    output_dir = "/media/ak/T7/tmp"
    if not os.path.exists(output_dir):
        print(f"The directory {output_dir} does not exist.")
    else:
        all_data = process_directory(rawDataFolder, symbol, output_dir)
