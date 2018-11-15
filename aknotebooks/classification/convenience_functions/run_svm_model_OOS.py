import numpy as np
import pandas as pd
from hsmm_core.data_utils import DataLoader, TradingHours
from hsmm_core.feature_spaces import hmm_features
from hsmm_core.hsmm_runner import HmmCalibration
import time
from hsmm_core.consts import InitialisationMethod
from hsmm_core.data_utils import TradingHours, DataLoader
from hsmm_core.labelling import DataLabellingSimple
from hsmm_core.consts import ThresholdMethod, LabellingChoice
import os


###-files-###

os.getenv('FINANCE_DATA')
ticker= 'SYNT_2states'
synt_data=os.listdir(os.path.join(os.getenv('FINANCE_DATA'), ticker)) #list of files with all the synthetic data
ticker_dir = os.path.join(os.getenv('FINANCE_DATA'),ticker)
synt_data_files = os.listdir(os.path.join(os.getenv('FINANCE_DATA'), ticker))
synt_data_dates=[os.path.splitext(file_name)[0] for file_name in synt_data_files]
synt_data_dates.sort()#sort the dates. no special reason makes it cleaner to play with the indexation
# test_file_name =os.path.join(ticker_dir,synt_data[0])


def fwd_dates(_dates_list, _key_date):

        # returns a list of dates that are forward from the key_date
    fwd_dates_list = [i for i in _dates_list if i > _key_date]

    return fwd_dates_list

###init files###
###hmm specification ###
n_hidden_states = 2

data_loader_init = {
    'trading_hours_filter': TradingHours.only_mkt_hours,
}


init_params = {
    'obs_model_name': 'ExpIndMixDiracGauss',
    'em_obs_init_method': InitialisationMethod.cluster,
    'em_hidden_init_method': InitialisationMethod.uniform,
    'no_hidden_states': n_hidden_states,
    'update_tag': 'tpsml'
}

window = 25
threshold = 0.1

labelling_method_params = [{

    'labelling_method': LabellingChoice.price_move_in_window,
    'rolling_window': window,
    # Uncomment below if you want to check a price move only above a certain level
    'updown_threshold': threshold,  # this is multiplied by 100
    'threshold_method': ThresholdMethod.arbitrary,
}]


sd = synt_data_dates[0]
ed = synt_data_dates[-1]

data_loader = DataLoader(**data_loader_init)

data_loader_hash = data_loader.data_loader_hash()

raw_data = data_loader.load_trades_data(ticker, start_date=sd, end_date=ed)

hmm_calibration_engine = HmmCalibration(init_params=init_params)

for label_init in labelling_method_params:
    labeller = DataLabellingSimple(label_init)
    the_label = labeller.get_label_name()
    labeller.label_training_data(raw_data)

print "ok"
