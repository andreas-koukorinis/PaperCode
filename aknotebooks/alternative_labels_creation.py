from hsmm_core.data_utils import TradingHours, DataLoader
from hsmm_core.labelling import DataLabellingSimple
from hsmm_core.consts import ThresholdMethod, LabellingChoice
import datetime as dt
import pickle
import os
import pandas as pd
import collections
import fileutils as fileutils


def get_files_for_ticker(ticker):
    """
    :return: Return all file names available for ticker, sort by name which is date
    """
    fin_data_dir = os.path.join("/media/ak/DataOnly", 'Data')
    total_path = os.path.join(fin_data_dir, ticker)
    # os.path.join(total_path, f)
    files_for_ticker = [f.split('.')[0] for f in os.listdir(total_path)
                        if (os.path.isfile(os.path.join(total_path, f)) and '.csv' in f and 'lock' not in f)]

    files_for_ticker.sort()
    return files_for_ticker


trading_hours_filter = TradingHours.only_mkt_hours
labelling_method_params = [

    {
        'labelling_method': LabellingChoice.ma_in_window,
        'rolling_window': 25,
        # Uncomment below if you want to check a price move only above a certain level
        'updown_threshold': 0.0001,
        'threshold_method': ThresholdMethod.arbitrary, },
    {
        'labelling_method': LabellingChoice.ma_in_window,
        'rolling_window': 15,
        # Uncomment below if you want to check a price move only above a certain level
        'updown_threshold': 0.0002,
        'threshold_method': ThresholdMethod.arbitrary, },
    {
        'labelling_method': LabellingChoice.price_move_in_window,
        'rolling_window': 2,
        # Uncomment below if you want to check a price move only above a certain level
        'updown_threshold': 0.005,
        'threshold_method': ThresholdMethod.arbitrary, }
    ,
    {
        'labelling_method': LabellingChoice.ep_in_window,
        'rolling_window': 9,
        # Uncomment below if you want to check a price move only above a certain level
        'updown_threshold': 0.0008,
        'threshold_method': ThresholdMethod.arbitrary, },
    {
        'labelling_method': LabellingChoice.ma_in_window,
        'rolling_window': 10,
        # Uncomment below if you want to check a price move only above a certain level
        'updown_threshold': 0.0018}]

if __name__ == '__main__':
    listDataPath = os.listdir(fileutils.data_path)
    symbols_listDataPath = [s for s in listDataPath if s.endswith('.L') or s.endswith('.I')]
    print(symbols_listDataPath)
    list_of_Labels_Symbols = os.listdir(fileutils.labels_path)
    print("A bunch of symbols from the listdir above:", list_of_Labels_Symbols)
    LabelsPathsNames = ['Labels', 'LabelsAlternateOne', 'LabelsAlternateTwo', 'LabelsAlternateThree', 'LabelsAlternateFour', 'LabelsAlternateFive']
    data_location_folder = '/media/ak/My Passport/Data/FinDataReal/'
    passport = '/media/ak/My Passport/Data/FinDataReal/'
    locations_dict = collections.OrderedDict()
    for ticker in list_of_Labels_Symbols :
        try:
            symboldateslist = get_files_for_ticker(str(ticker))
            print(ticker)
            sd = symboldateslist[0]
            ed = symboldateslist[-1]

            trading_hours_filter = TradingHours.only_mkt_hours

            data_loader = DataLoader(trading_hours_filter)

            data = data_loader.load_trades_data(ticker, start_date=sd, end_date=ed)
            for label_init in labelling_method_params:
                labeller = DataLabellingSimple(label_init)
                the_label = labeller.get_label_name()
                labeller.label_training_data(data)
                filename = os.path.join(passport, ticker.split(".")[0]+ '_AlternateLabels.pickle')
                print(filename)
                locations_dict[ticker][str(label_init)] = filename
                with open(filename, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    filename.close()

            print("ok")
        except Exception:
            continue
    dict_file = open(os.path.join(data_location_folder,'locations_dict.pkl'), 'wb')
    pickle.dump(locations_dict, dict_file)
    dict_file.close()

