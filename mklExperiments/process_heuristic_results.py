import os
import pickle
import pandas as pd
import sys
import sys
from memory_profiler import profile
import pandas as pd
import numpy as np
sys.path.append('/home/ak/Documents/Research/PaperCode/MultiKernelLearning')
sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import jsonpickle
from collections import defaultdict
from sklearn.svm import SVC
from collections import OrderedDict
import os
from sklearn.metrics import accuracy_score
import torch


from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures
import multiprocessing

def open_pickle_filepath(pickle_file):
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file

simple_mkl_path = '/media/ak/DataOnly/ExperimentCommonLocs/MKLOOSPredictions/SimpleMKL/'

#processed results path
processed_location = '/media/ak/DataOnly/ExperimentCommonLocs/MKLOOSProcessed'
heuristic_path = '/media/ak/DataOnly/ExperimentCommonLocs/MKLOOSPredictions/Heuristic/'
files_location = heuristic_path
symbols = sorted(os.listdir(files_location))

if __name__ == '__main__':
    symbols = os.listdir(heuristic_path)

    chunks_dict= {}
    list_of_dfs = list()
    for symbolIDX in [7]: #do 6: BLT
        print('processing symbol:', symbolIDX, sorted(symbols)[symbolIDX])

        files_location = os.path.join(heuristic_path, sorted(symbols)[symbolIDX])
        print(files_location)
        print('total files for this symbol: ', len(files_location))
        print(os.path.isdir((files_location)))
        for label_idx in [3]:
            print('and now doing label: ', label_idx)
            try:

                symbol_label_files = [f for f in os.listdir(files_location) if str('_Label_idx:_'+str(label_idx)+'_') in f]
                print('total files for the symbol/label:', len(symbol_label_files)) # total how many you have and how many for a specific label

                number_of_files = len(symbol_label_files)
                print('and for this label', number_of_files)
                for file_idx in range(0, number_of_files):
                    print(file_idx)
                    file_to_load_path = os.path.join(files_location, symbol_label_files[file_idx])
                    if (os.path.isfile(file_to_load_path)):
                        try:
                            if os.path.getsize(file_to_load_path)> 0:
                                # extract the key - these are the symbol feature path keys in the  main code

                                file_dict = open_pickle_filepath(file_to_load_path)

                                file_keys = list(file_dict.keys()) # keys

                                chunks_dict[file_keys[0]] = list(file_dict[file_keys[0]].keys()) # these are the chunks dict
                                for key_item in list(file_dict[file_keys[0]].keys()):
                                    list_of_dfs.append(pd.DataFrame(file_dict[file_keys[0]][key_item]))

                                concat_df = pd.concat(list_of_dfs, axis =1).median(axis =1 )
                                title = sorted(symbols)[symbolIDX] +"_Label_"+str(label_idx)+'_'+str(file_idx)+'_median_dfs_heuristic.csv'
                                location_to_save=os.path.join(processed_location, sorted(symbols)[symbolIDX], 'Heuristic', title)
                                concat_df.to_csv(location_to_save, index = True)
                                print('and now saved :', location_to_save)
                        except (OSError, ValueError, TypeError, EOFError,FileNotFoundError, IndexError,PermissionError, FileNotFoundError, RuntimeError):

                            continue
                    else:
                        pass
            except (FileNotFoundError):
                pass


