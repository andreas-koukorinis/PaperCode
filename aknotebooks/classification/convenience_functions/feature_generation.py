import numpy as np
import pandas as pd
import sys
from scipy.interpolate import interp1d
from scipy import interpolate

# sys.path.append('/home/kman/Research/hsmm/')
sys.path.append('/home/ak/Documents/Research/hsmm/hsmm/')

from hsmm_core.hmm import *
from hsmm_core.hmm_utils import load_data
from test_hmm.test_utils import generate_sample_paths_independent_sets
from hsmm_core.observation_models import *
from hsmm_core.hmm import *
from hsmm_core.prediction_engines import *
from hsmm_core.hmm_utils import load_data
from hsmm_core.prediction_engines import *
import pickle

ticker = 'EEM'
_features_path= '/home/ak/Documents/features_test/' #directory to save data
#labelling methods for the data- these are used within the SVM- changing from simple to more complicated :cc_labels
labelling_methods = [
    {'method': dl_choices.control_chart,'cc_quantiles': [30, 70],'rolling_window': 5,},
    {'method': dl_choices.control_chart,'cc_quantiles': [20, 80],'rolling_window': 10,},
    {'method': dl_choices.simple_ep,'rolling_window': 5,
        'cmp_threshold': 0.05,  # Use this value to set a threshold in comparison, i.e. it does not suffice
        # to have P_t < P_{t + window} but  P_t < ( 1- cmp_threshold) * P_{t + window}
    },
]

labellers_ = []

for labelling_method in labelling_methods:
    method = labelling_method['method']

    if method is dl_choices.control_chart:
        # look for qunatile values, if not use default 10/90
        labellers_.append(data_labelling_cc(**labelling_method))
    else:
        labellers_.append(data_labelling_simple(**labelling_method))
#no_days: key component here
data_dic, data_sampling_tag = load_data(ticker, resample=None, no_of_days=20)

for labeller in labellers_:
    labeller.label_training_data(data_dic)

# Store data (serialize)
with open(_features_path+(ticker)+'_labelled_data.pickle', 'wb') as handle:
    pickle.dump(data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
zero_pc_threshold = 1e-5
n_hidden_states = 2
init_params = {"obs_model_params": {'epsilon': zero_pc_threshold,
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
prediction_params = {
    'labelling_methods':  labelling_methods,
    'feature_generation': feature_generation.one_model_to_one_fset,
}

prediction_eng = hs_prediction_engine(hmm_init_params=init_params,prediction_innit_params=prediction_params,no_parallel_procs=None)
#
# #generates features and models
features, _models_ = prediction_eng.train_model_data(ticker, data_dic, 'resample__', force_hmm_recalc=False)

_label_one_path ='/home/ak/Documents/Data/features_models/Rolling_5_updown_10_simple_ep/'
# Store data (serialize)
with open(str(_features_path)+(ticker)+'_features.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
