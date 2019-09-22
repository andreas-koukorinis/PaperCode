import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import BDay
from hsmm_core.utils import mc_limiting_distribution, states_from_limit_dist
from hsmm_core.observation_models import ExpIndMixDiracGauss
from hsmm_core.data_utils import TradingHours, DataLoader
from hsmm_core.labelling import DataLabellingSimple
from hsmm_core.consts import ThresholdMethod, LabellingChoice
import datetime as dt
from pandas.tseries.offsets import BDay
from hsmm_core.utils import mc_limiting_distribution, states_from_limit_dist
from hsmm_core.observation_models import ExpIndMixDiracGauss
from hsmm_core.feature_spaces import hmm_features
from hsmm_core.hsmm_runner import HmmCalibration

from hsmm_core.hmm import hmm_impl

# from hsmm_core.data_utils import DataLoader, TradingHours
# from hsmm_core.labelling import DataLabellingSimple
# from hsmm_core.consts import ThresholdMethod, LabellingChoice
# import pickle
# from hsmm_core.consts import InitialisationMethod
# import datetime as dt

# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF


from hsmm_core.data_utils import DataLoader, TradingHours
from hsmm_core.labelling import DataLabellingSimple
from hsmm_core.consts import ThresholdMethod, LabellingChoice
import pickle
from hsmm_core.consts import InitialisationMethod
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

###-functions-###

def fwd_dates(dates_list, key_date):
    # returns a list of dates that are forward from the key_date
    fwd_dates_list = [i for i in dates_list if i > key_date]
    return fwd_dates_list


def remove_nans(features_tuple, labels, idx=1):
    # not the cleanest but useful
    # function to clean up nans as I seem to use it a lot, so better to have one function
    # combines the features and labels and removes rows with nans across so we dont lose the ordering
    # returns features and labels
    features_df = pd.concat([features_tuple[0], features_tuple[1], features_tuple[2], \
                             features_tuple[3]], axis=1, sort=False)
    labels_only = labels.drop(columns=['ReturnTradedPrice', 'Duration', 'states', 'TradedTime',
                                       'TradedPrice'], axis=1)
    df_concat = pd.concat([features_df, labels_only.iloc[:, 0:idx]], axis=1, sort='False')
    # only using 1st set of labels- but we can re-write this a bit
    df_x_nan = df_concat.dropna()  # dropping all nans
    label_column_loc_ = df_x_nan.shape[1] - 1  # location of labels column in the clean df
    labels_ = df_x_nan.iloc[:, label_column_loc_:label_column_loc_ + 1]  # keep pure labels
    features_ = df_x_nan.drop(df_x_nan.columns[label_column_loc_], axis=1)  # keeping the features only

    return features_, labels_ #return features and labels in the X,y order that scikit takes the input


def prec_recall_report(y_true, y_predict):
    # function to ge the sci-kit learn classification metrics into a pretty DF for csv!
    report = pd.DataFrame(list(precision_recall_fscore_support(y_true, y_predict)),
                          index=['Precision', 'Recall', 'F1-score', 'Support']).T
    # Now add the 'Avg/Total' row
    report.loc['Avg/Total', :] = precision_recall_fscore_support(y_true, y_predict, average='weighted')
    report.loc['Avg/Total', 'Support'] = report['Support'].sum()
    return report


if __name__ == '__main__' :
    #  setting locations
    ticker_list= [ 'YM1_Comdty', 'VXX_US_Equity', 'XM1_Comdty', 'TU1_Comdty', 'US1_Comdty',
    'FB1_Comdty', 'KE1_Comdty', 'JB1_Comdty', 'VIX_Index', 'TY1_Comdty',
    'FV1_Comdty', 'DU1_Comdty', 'XM1_Comdty', 'OE1_Comdty', 'RX1_Comdty', 'G_1_Comdty',
    'US1_Comdty']

    no_states = 3

    hmm_init = {
        'obs_model_name': 'CensoredExpIndMixDiracGauss',
        'em_obs_init_method': InitialisationMethod.cluster,
        'em_hidden_init_method': InitialisationMethod.uniform,
        'no_hidden_states': no_states,
        'update_tag': 'tpsml'
    }
    for ticker in ticker_list:

        data_dir = os.getenv('FINANCE_DATA') #main directory
        # features_path = os.path.join(os.path.expanduser("~"), 'Data/features_models/features/') #where features are saved
        # labels_path = os.path.join(os.path.expanduser("~"), 'Data/features_models/labels') #where labels are saved
        models_path = os.path.join(data_dir, 'features_models/models/')
        features_path = os.path.join(data_dir, 'features_models/features/')
        labels_path = os.path.join(data_dir, 'features_models/labels/')
        ticker_labels_path = os.path.join(labels_path, ticker + '/NON_DIRECTIONAL')
        ticker_models_path = os.path.join(models_path, ticker + '/SINGLE_KERNEL')
        ticker_hmm_path= os.path.join(models_path, ticker + '/HMM')
        ticker_features_path = os.path.join(features_path, ticker + '/MODEL_BASED/')

        if not os.path.exists(os.path.join(data_dir, ticker)):
            os.makedirs(os.path.join(data_dir, ticker))

        if not os.path.exists(ticker_labels_path):
            os.makedirs(ticker_labels_path)

        if not os.path.exists(ticker_models_path):
            os.makedirs(ticker_models_path)

        if not os.path.exists(ticker_hmm_path):
            os.makedirs(ticker_hmm_path)

        if not os.path.exists(ticker_features_path):
            os.makedirs(ticker_features_path)

        labels_list = os.listdir(ticker_labels_path)

        print "making paths-now fit hmms (and save them)"

        trading_hours_filter = TradingHours.only_mkt_hours

        data_loader = DataLoader(trading_hours_filter)#for vol_clock---> ccy_bar=1000)
        # keep the hash of the data loader to uniquely identify how the data was loaded ( perhaps a dollar clock was
        # used), as this affects the calibration of the hmm
        data_loader_hash = data_loader.data_loader_hash()

        data = data_loader.load_trades_data(ticker)
        #
        hmm_calibration_engine = HmmCalibration(init_params=hmm_init)
        hmm_calibration_engine.run_calibration_all_data(ticker, data, data_loader_hash,
                                                        force_recalc=False, use_multiprocessing=False,
                                                        n_processes=2)

        # Create the hmm feature engine and for every change the hmm model in the features engine
        features_engine = hmm_features()
        # #
    # # Create Labels ###

        window = 5
        threshold = 0.001

        labelling_method_params = [{

            'labelling_method': LabellingChoice.price_move_in_window,
            'rolling_window': window,
            # Uncomment below if you want to check a price move only above a certain level
            'updown_threshold': threshold,  # this is multiplied by 100
            'threshold_method': ThresholdMethod.arbitrary,
        }]

        for label_init in labelling_method_params:
            print label_init
            labeller = DataLabellingSimple(label_init)
            labeller.label_training_data(data)

        for date, date_data in data.iteritems():
            date_data.to_csv(os.path.join(ticker_labels_path, str(date)+'.csv')) ##savind data and labels together now##

    #
    #     # # # Iterate through dates, load the stored hmm, make the features_engine point at that, and use the labels
    #     # # # already added as columns to the data data frames
        now = dt.datetime.now()  # day we use to generate features

        for date, date_data in data.iteritems():
            new_path = os.path.join(ticker_features_path, str(date)) #create a new path inside features for each date
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            stored_hmm, _ = hmm_calibration_engine.get_calibrated_hmm(ticker, date, data_loader_hash) #model, hash and date
            stored_hmm_file = "_".join(
                (str(ticker),"model_date_:", str(date), str(no_states), "states", "stored_hmm", now.strftime('%Y%m%d'), ".pickle"))
            pickle.dump(stored_hmm, open(os.path.join(ticker_hmm_path, stored_hmm_file), 'wb')) #storing the hmm model using todays date

            ##get the right features using forward dates###
            features_engine.hmm = stored_hmm #engine for features which is essentialy the fitted hmm on a specific date
            list_fwd_dates = fwd_dates(dates_list=list(data.keys()), key_date=date) #take all the dates ahead of your hmm_date

            for fwd_date in list_fwd_dates:
                features_load = features_engine.generate_features(data[fwd_date]) #get features using the model but fwd
                features_file = "_".join((str(ticker), str(no_states), "states", "features_date:", str(fwd_date),"now:",
                                          now.strftime('%Y%m%d'), ".pickle"))
                print(features_file)
                pickle.dump(features_load, open(os.path.join(new_path, features_file), 'wb'))
