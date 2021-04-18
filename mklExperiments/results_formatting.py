# append all the necessary paths
import sys
sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
sys.path.append('/home/ak/Documents/Research/PaperCode/MultiKernelLearning')
# %reload_ext memory_profiler
import new_alternate_single_svm as nalsvm
# import mkl_data_processing as mkl_dp
# import mkl_model_cross_validation as mkl_cv
import os
import numpy as np
import pickle as pickle
import pandas as pd


###########    Files   #############
passportDrive ='/media/ak/My Passport'
dataOnlyDrive = '/media/ak/DataOnly/'
finDataFolder = os.path.join(dataOnlyDrive, 'FinDataReal')
experimentDataPassportDrive = os.path.join(passportDrive, 'ExperimentData')
alternativeExperimentPath = os.path.join(experimentDataPassportDrive,'AlternateLabelExperimentPath')
oosResultsPath = '/media/ak/My Passport/Data/FinDataReal/JointLocationsAlternateDataClean/OOS_Results'

def load_pickled_in_filename(file):
    # load a simple pickled file and return it. its a bit different to the method used for the dictionary as this
    # is pure Python 3.x
    pickle_in = open(file, 'rb')
    return pickle.load(pickle_in)


def mean_results(experimentsList, DateIdx):
    pkled_file_to_load = os.path.join(symbolPath, experimentsList[DateIdx])
    pkled_dict = load_pickled_in_filename(pkled_file_to_load)[list(load_pickled_in_filename(pkled_file_to_load).keys())[0]]
    return pd.DataFrame.from_dict(pkled_dict).mean(axis=1)


def symbol_results(labelone_mean_results_list, labelthree_mean_results_list, labelfour_mean_results_list,
                   labelfive_mean_results_list):
    symbol_results = dict()
    symbol_results['Label One'] = pd.DataFrame(labelone_mean_results_list).median(axis=0)
    symbol_results['Label Three'] = pd.DataFrame(labelthree_mean_results_list).median(axis=0)
    symbol_results['Label Five'] = pd.DataFrame(labelfive_mean_results_list).median(axis=0)
    symbol_results['Label Four'] = pd.DataFrame(labelfour_mean_results_list).median(axis=0)

    return symbol_results


if __name__ == '__main__':
    for sumbolPathIdx in range(0, 26):
        print(sumbolPathIdx)
        symbols_with_experiments = os.listdir(alternativeExperimentPath)
        symbolPath = os.path.join(alternativeExperimentPath, symbols_with_experiments[sumbolPathIdx])
        print(symbols_with_experiments[sumbolPathIdx])
        print(len(os.listdir(symbolPath)))
        files = os.listdir(symbolPath)
        LabelFiveExperiments = [f for f in files if str('Five') in f]
        LabelFourExperiments = [f for f in files if str('Four') in f]
        LabelThreeExperiments = [f for f in files if str('Three') in f]
        # LabelTwoExperiments = [f for f in files if str('Two') in f]
        LabelOneExperiments = [f for f in files if str('One') in f]

        # 5
        labelfive_mean_results_list = list()
        for i in range(len(LabelFiveExperiments) - 1):
            labelfive_mean_results_list.append(mean_results(LabelFiveExperiments, i))

        labelfour_mean_results_list = list()
        # 4
        for i in range(len(LabelFourExperiments) - 1):
            labelfour_mean_results_list.append(mean_results(LabelFourExperiments, i))

        labelthree_mean_results_list = list()
        # 3
        for i in range(len(LabelThreeExperiments) - 1):
            labelthree_mean_results_list.append(mean_results(LabelThreeExperiments, i))
        # 1
        labelone_mean_results_list = list()
        for i in range(len(LabelOneExperiments) - 1):
            labelone_mean_results_list.append(mean_results(LabelOneExperiments, i))

        # symbol_results = symbol_results(labelone_mean_results_list, labelthree_mean_results_list,
        #                                 labelfour_mean_results_list, labelfive_mean_results_list)
        #
        #
        #
        # print(pd.DataFrame.from_dict(symbol_results))