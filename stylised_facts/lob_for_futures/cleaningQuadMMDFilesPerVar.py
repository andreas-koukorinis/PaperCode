import os
import pandas as pd
import numpy as np
import pickle

def pickle_dump_obj_to_filename(destinationPath, symbol, fileName, obj):
    pickle_out_filename = os.path.join(destinationPath, "_".join((symbol, fileName)))
    with open(pickle_out_filename, 'wb') as pickle_out:
        pickle.dump(obj, pickle_out)
    print('saved', pickle_out_filename)

if __name__ == "__main__":
    path = '/media/ak/T71/August11th2022Experiments/QuadMMDOutputFiles/'
    alphaResults = 'processedAlphaResults'
    outputFolder = os.path.join(path, 'processedDataFrames')
    symbol = 'G_1'
    var_choice = 'tick'

    resultsFolder = os.path.join(path, alphaResults, symbol, var_choice)
    folder_files = os.listdir(resultsFolder)
    files = [f for f in sorted([f for f in folder_files if str(symbol) in f]) if str(var_choice) in f]

    path = resultsFolder
    dflist = list()
    for idx in range(0, len(files)):
        filePath = os.path.join(path, files[idx])
        dflist.append(pd.read_pickle(filePath))

    # Initialize an empty list to store the DataFrames
    df_list = []

    # Iterate over each dictionary in dflist
    for data_dict in dflist:
        # Extract the 'results' dictionary from the data_dict
        results_dict = data_dict['results']

        # Calculate medians for each key in the results dictionary
        medians = {key: np.median(value) if isinstance(value, np.ndarray) else value for key, value in
                   results_dict[data_dict['start_point']].items()}

        # Extract the 'XZ_test' dictionary from the results_dict
        xz_test_dict = results_dict[data_dict['start_point']]['XZ_test']

        # Combine the 'XZ_test' keys with 'XZ_test_' prefix
        xz_test_keys = ['XZ_test_' + key for key in xz_test_dict.keys()]

        # Add the 'XZ_test' keys to the medians dictionary
        medians.update(zip(xz_test_keys, xz_test_dict.values()))

        # Extract the 'YZ_test' dictionary from the results_dict
        yz_test_dict = results_dict[data_dict['start_point']]['YZ_test']

        # Combine the 'YZ_test' keys with 'YZ_test_' prefix
        yz_test_keys = ['YZ_test_' + key for key in yz_test_dict.keys()]

        # Add the 'YZ_test' keys to the medians dictionary
        medians.update(zip(yz_test_keys, yz_test_dict.values()))

        # Replace the values of XZ_test_list_permuted_mmd2 and YZ_test_list_permuted_mmd2 with their medians
        medians['XZ_test_list_permuted_mmd2'] = np.median(xz_test_dict['list_permuted_mmd2'])
        medians['YZ_test_list_permuted_mmd2'] = np.median(yz_test_dict['list_permuted_mmd2'])

        # Create a DataFrame with a single row
        df = pd.DataFrame([medians])

        # Create a multi-index with the four keys as levels
        index = pd.MultiIndex.from_tuples(
            [(data_dict['start_point'], data_dict['end_point'], data_dict['shift'], data_dict['window'])],
            names=['start_point', 'end_point', 'shift', 'window'])

        # Set the multi-index for the DataFrame
        df.index = index

        # Drop the XZ_test and YZ_test columns
        df = df.drop(columns=['XZ_test', 'YZ_test'])

        # Append the DataFrame to the list
        df_list.append(df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list)
    window_summary = combined_df.pivot_table(index='window', aggfunc='mean')

    # Display the combined DataFrame
    print(window_summary)
    windowFileName = "_".join(( var_choice, 'processedAlphaResults_window_DF.pkl'))
    print(windowFileName )
    pickle_dump_obj_to_filename(outputFolder, symbol, windowFileName, window_summary)


