import numpy as np
import pandas as pd
import os
import sys

from pandas.tseries.offsets import BDay
from hsmm_core.hmm import hmm_engine
from hsmm_core.observation_models import ExpIndMixDiracGauss

from hsmm_core.data_utils import load_data, TradingHours
from hsmm_core.labelling import DataLabellingSimple
from hsmm_core.consts import ThresholdMethod, LabellingChoice
import datetime as dt

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

ticker = 'SYNT_2states'
trading_hours_filter = TradingHours.all_trading_day
data = load_data(ticker, which_trading_hours=trading_hours_filter)
labels_path= '/home/ak/Data/features_models/labels'

ticker_labels_path = os.path.join(labels_path,ticker)


labelling_method_params = [
# {
#     'labelling_method': LabellingChoice.simple_ep_abs,
#     'rolling_window': 10,
#     'threshold_method': ThresholdMethod.historical_vol_ma,
#     'thrshd_vol_window': 5,
#     'updown_thrshd_vol_pct': 10.,
#     'trading_hours_filter': trading_hours_filter,
# },
{

# ticker_labels_path = os.path.join(labels_path,ticker+'/NON_DIRECTIONAL')

labelling_method_params = [{

    'labelling_method': LabellingChoice.price_move_in_window,
    'rolling_window': 2,
    # Uncomment below if you want to check a price move only above a certain level
    'updown_threshold': 0.1,

    'threshold_method': ThresholdMethod.arbitrary,
}]

# for label_init in labelling_method_params:
#     labeller = DataLabellingSimple(label_init)
#     labeller.label_training_data(data)
#
# keys_ = data.keys()

for key_, _ in enumerate(keys_):
    data[keys_[key_]].to_csv(ticker_labels_path+'/'+str(keys_[key_])+'.csv', index=False)

#write this in a more pythonic way

