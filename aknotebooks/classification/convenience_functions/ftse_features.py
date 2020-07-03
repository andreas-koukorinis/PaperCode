import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
sys.path.append('/home/ak/Documents/Research/hsmm/hsmm/')
#sys.path.append('/home/kman/Research/hsmm')
_features_path= '/home/ak/Documents/Data/features_models/' #directory to save data
_path= '/home/ak/Documents/Data/'
_ftse = os.path.join(_path,'FTSE')
###HMM requirements
from hsmm_core.observation_models import *
from hsmm_core.hmm import *
from hsmm_core.hmm_utils import load_data, TradingHours
from hsmm_core.prediction_engines import *
###
##useful functions##
def listdirs(folder): #return only directories from a master folder
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
###

symbols_list= os.listdir(_ftse)

for _no, _ in enumerate(symbols_list):
     ticker = symbols_list[_no]
     ticker_path = os.path.join(_ftse,symbols_list[_no])
     print ticker
     print ticker_path
     no_days =len(os.path.join(_ftse,symbols_list[_no]))
     print no_days

n_hidden_states = 3

zero_pc_threshold = 1e-5
init_params = {
    "obs_model_params": {
                                'epsilon': zero_pc_threshold,
                                'obs_model_name': 'ExpUniGauss',
                                'init_params': {'epsilon': zero_pc_threshold},
                                'em_init_method': initialization_method.cluster

    },
    "hidden_model_params": {
                                'no_hidden_states': n_hidden_states,
                                # 'pi':startprob,
                                # 'tpm': transmat
                                'em_init_method': initialization_method.uniform
    },
    "update_tag": 'tpsml'
}

# Look out for the flag which_trading_hours. Possible values are:
# TradingHours.pre_open_hours, TradingHours.only_mkt_hours, TradingHours.all_trading_day
# with the obvious interpretation
data_dic, data_sampling_tag =load_data(ticker,
                                       no_of_days=no_days,
                                       #file_name=['20120217.csv'],
                                       load_columns=['TradedTime', 'Duration','TradedPrice', 'ReturnTradedPrice'],
                                       which_trading_hours=TradingHours.only_mkt_hours)
