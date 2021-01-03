import sys

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import os
import pandas as pd
import numpy as np
import pickle as pickle

# paths
oos_experiment_path = '/media/ak/My Passport/ExperimentData/AlternateLabelExperimentPath'
processed_data_for_experiments_path = '/media/ak/My Passport/ExperimentData/ProcessedDataForExperiments'
oos_experiment_symbol_folders = sorted([f for f in os.listdir(oos_experiment_path) if str('pkl') not in f])


# functions

def load_pickled_in_filename(file):
    # load a simple pickled file and return it. its a bit different to the method used for the dictionary as this
    # is pure Python 3.x
    pickle_in = open(file, 'rb')
    return pickle.load(pickle_in)


def get_table_from_dict(dict_list, label):
    df = concatenate_all_oos_per_label(dict_list)
    df_w_label = pd.DataFrame(df.mean(axis=1)).rename(columns={0: label})
    return df_w_label


def concatenate_all_oos_per_label(dict_list):
    # the barani solution
    """

    :param dict_list: takes a list o dictionaries for the way the formatted OOS results are
    :return: a clean concatenated dataframe, the values of the row index across all dataframes must be the same!
    """
    list_of_dfs = [pd.DataFrame(l) for l in dict_list]
    combined_results = pd.concat([pd.DataFrame(l) for l in dict_list], axis=1)
    combined_results
    return combined_results


def filename_for_df_to_csv(label, symbol):
    """
    returns a full path to store a dataframe in the form of a csv
    :param label: which labels data are we consolidating
    :param symbol: which symbol
    :return: returns a full path
    """
    filename = "_".join((symbol, label, "consolidatedResults.csv"))
    filename_for_df = os.path.join(processed_data_for_experiments_path, filename)
    return filename_for_df


def create_mean_list_w_correct_label(df, label_string):
    df = pd.DataFrame(df.mean(axis=1))
    df = df.rename(columns={df.columns[0]: label_string}, inplace=True)
    return df


def labels_files(folder_files):
    LabelsAlternateFive_symbol_files = [f for f in folder_files if str('LabelsAlternateFive') in f]
    LabelsAlternateFour_symbol_files = [f for f in folder_files if str('LabelsAlternateFour') in f]
    LabelsAlternateThree_symbol_files = [f for f in folder_files if str('LabelsAlternateThree') in f]
    LabelsAlternateTwo_symbol_files = [f for f in folder_files if str('LabelsAlternateTwo') in f]
    LabelsAlternateOne_symbol_files = [f for f in folder_files if str('LabelsAlternateOne') in f]
    labels_files_list = [LabelsAlternateOne_symbol_files, LabelsAlternateTwo_symbol_files,
                         LabelsAlternateThree_symbol_files, LabelsAlternateFour_symbol_files,
                         LabelsAlternateFive_symbol_files]
    return labels_files_list


if __name__ == "__main__":
    oos_symbol_idx = 0  # this corresponds to a symbol
    for oos_symbol_idx in range(27):
        symbol = oos_experiment_symbol_folders[oos_symbol_idx]
        print(symbol)

        oos_symbol_folder_files = [f for f in os.listdir(os.path.join(oos_experiment_path, symbol)) if
                                   str('_OOS_results') in f]
        # this is a structured list of lists.
        # each one is a specified one with output.
        # some may be empty so need to adjust for that - SOME REFACTORING NEEDED
        oos_symbol_list_of_files_list = labels_files(oos_symbol_folder_files)

        # so i need a per label filter - so now pick one label # pick one label
        for per_alternate_label_filter_Idx in range(4):
            # using the above filter-idx i pick one list - all these should be against the same label
            LabelsAlternate_symbol_files = oos_symbol_list_of_files_list[per_alternate_label_filter_Idx]
            number_model_files = len(LabelsAlternate_symbol_files)
            if number_model_files > 0:  # if the list is not empty
                label = LabelsAlternate_symbol_files[0].split("_")[1]  # picke A file to extract the label
                # use this for the loop and for the if condition

                # get all the files from that list
                print(label)

                dict_list = []  # will store all the results of the concatenated data
                for file in LabelsAlternate_symbol_files:
                    pickle_file_location = os.path.join(oos_experiment_path, symbol, file)
                    df_w_Data = load_pickled_in_filename(pickle_file_location)
                    print(df_w_Data)
                    ans = {label.split("_")[1]: {metric: np.mean([value[metric] for value in values.values()]) for metric in
                                                 values[list(values)[0]].keys()} for label, values in df_w_Data.items()}
                    # print(ans)
                    dict_list.append(ans)

                print(dict_list)

                df_to_store = pd.DataFrame((get_table_from_dict(dict_list, label)))
                filename = filename_for_df_to_csv(label, symbol)
                df_to_store.to_csv(filename)
                print('saved')


            else:
                print('empty label file')
                pass
    pass
