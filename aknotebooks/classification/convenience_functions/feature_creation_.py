import sys
sys.path.append('/home/ak/Documents/Research/hsmm/')
from hsmm_core.prediction_engines import *
from hsmm_core.data_utils import load_data, TradingHours
import pickle
from hsmm_core.consts import ThresholdMethod, LabellingChoice

symbols_ftse= ['SYNT_2states']
_features_path='/home/ak/Data/features_models/features/'


n_hidden_states = 2

init_params = {
    "obs_model_params": {
                                'obs_model_name': 'ExpIndMixDiracGauss',
                                'em_init_method': InitialisationMethod.cluster

    },
    "hidden_model_params": {
                                'no_hidden_states': n_hidden_states,
                                # 'pi':startprob,
                                # 'tpm': transmat
                                'em_init_method': InitialisationMethod.uniform
    },
    "update_tag": 'tpsml'
}

for ticker in symbols_ftse:

    print ticker

    labelling_method_params = {
        'labelling_method': LabellingChoice.simple_ep_pct,
        'rolling_window': 5,
        'updown_threshold': 10.,
        'threshold_method': ThresholdMethod.arbitrary,
    }


    prediction_params = {
        'labelling_method_params': labelling_method_params,
        'feature_generation': FeatureGeneration.one_model_to_one_fset,
    }

    trading_hours = TradingHours.all_trading_day

    data_dic = load_data(ticker)#, which_trading_hours=TradingHours.all_trading_day, is_synthetic=True)
    prediction_eng = hs_prediction_engine(hmm_init_params=init_params,
                                          prediction_innit_params=prediction_params,
                                          no_parallel_procs=None)
    features, _ = prediction_eng.train_model_data(ticker, data_dic, trading_hours, force_hmm_recalc=True)

    with open(_features_path+(ticker)+'_features.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
