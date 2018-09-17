import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/ak/Documents/Research/master/hsmm_core/')
sys.path.append('/home/ak/Documents/Research/master/')

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
labels_path= '/home/ak/Documents/Data/features_models/labels'
ticker_labels_path = os.path.join(labels_path,ticker)
# sd = '20171002'
# ed = dt.datetime(2017, 12, 30)
# for rol_window in range(10,13):
labelling_method_params = [
{
    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 5,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 15,
    'updown_thrshd_vol_pct': 10.,
    'trading_hours_filter': trading_hours_filter,
},
{

    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 2,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 5,
    'updown_thrshd_vol_pct': 10.,
    'trading_hours_filter': trading_hours_filter,
},
{
    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 5,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 5,
    'updown_thrshd_vol_pct': 10.,
    'trading_hours_filter': trading_hours_filter,
},
{
    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 7,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 7,
    'updown_thrshd_vol_pct': 10.,
    'trading_hours_filter': trading_hours_filter,
},
{
    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 9,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 9,
    'updown_thrshd_vol_pct': 5.,
    'trading_hours_filter': trading_hours_filter,
},
{
    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 5,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 5,
    'updown_thrshd_vol_pct': 5.,
    'trading_hours_filter': trading_hours_filter,
},
{
    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 5,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 5,
    'updown_thrshd_vol_pct': 15.,
    'trading_hours_filter': trading_hours_filter,
},
    {
    'labelling_method': LabellingChoice.simple_ep_abs,
    'rolling_window': 5,
    'threshold_method': ThresholdMethod.historical_vol_ma,
    'thrshd_vol_window': 5,
    'updown_thrshd_vol_pct': 25.,
    'trading_hours_filter': trading_hours_filter,
},
{
    'labelling_method': LabellingChoice.simple_ep_pct,
    'rolling_window': 5,
    'updown_threshold': 10.,
    'threshold_method': ThresholdMethod.arbitrary,
},
{
    'labelling_method': LabellingChoice.simple_ep_pct,
    'rolling_window': 10,
    'updown_threshold': 10.,
    'threshold_method': ThresholdMethod.arbitrary,
},
{
    'labelling_method': LabellingChoice.simple_ep_pct,
    'rolling_window': 15,
    'updown_threshold': 5.,
    'threshold_method': ThresholdMethod.arbitrary,
}]

# some date bounds, mix ddata types for dates

# for label_init in labelling_method_params:
#     print label_init.get('labelling_method', None)


for label_init in labelling_method_params:
    labeller = DataLabellingSimple(
                                    labelling_method=label_init.get('labelling_method', None),
                                    trading_hours_filter=trading_hours_filter ,
                                    thrshd_method=label_init.get('threshold_method', None),
                                    rolling_window=label_init.get('rolling_window', None),
                                    updown_thrshd=label_init.get('updown_threshold', None),
                                    updown_thrshd_vol_pct=label_init.get('updown_thrshd_vol_pct', None),
                                    thrshd_vol_window=label_init.get('thrshd_vol_window', None)
    )
    print (labeller.get_label_name())
    labeller.label_training_data(data)

keys_=data.keys()

for key_,_ in enumerate(keys_):
    data[keys_[key_]].to_csv(ticker_labels_path+'/'+str(keys_[key_])+'.csv',index=False)
#write this in a more pythonic way

