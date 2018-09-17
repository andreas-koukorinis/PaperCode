import numpy as np
import sys
sys.path.append('/home/ak/Documents/Research/hsmm/')
from hsmm_core.hmm import hmm_engine
from hsmm_core.statistics import *
from hsmm_core.observation_models import *
import pandas as pd
from pandas.tseries.offsets import BDay
import os
import datetime as dt

####
finance_data =os.getenv('FINANCE_DATA')
ticker ='mySYNT_3states'
target_dir =os.path.join(finance_data, ticker)
####
def biz_days_list(num_days_):
    end_date = pd.datetime.today()  # datetime.datetime.today()
    start_date = (end_date - BDay(num_days_)).to_pydatetime()
    date_list = [((start_date + BDay(x)).to_pydatetime()).strftime('%Y%m%d') for x in range(0, num_days_)]
    return date_list

def timestamps_(sequence_length):
    dateTimeA = dt.datetime.combine(dt.date.today(), dt.time(8, 00, 00))
    dateTimeB = dt.datetime.combine(dt.date.today(), dt.time(16, 30, 00))
    delta_ = dt.timedelta(days=0,seconds=np.abs((dateTimeA - dateTimeB).total_seconds())/ sequence_length,microseconds=1.06)
    timestamps_= [(dateTimeA+ delta_*x).strftime('%H:%M:%S.%f') for x in range (0,sequence_length)]
    return timestamps_


n_hidden_states = 3

obs_model_init = {
    'sigmas': np.array([0.5, 0.023, 0.45]),
    'lambdas': np.array([1., 0.01, 0.2]),
    'weights': np.array([0.2, 0.4, 0.4]),
}

# sigmas = [0.1, 4]
# lambdas = [0.5, 5]
# weights = [0.1, 0.1]

tpm = np.array([[0.6, 0.2, 0.2],[0.6, 0.2, 0.2],[0.6, 0.2, 0.2]]) #,
startprob = np.array([0.5, 0.2, 0.3]) #, 0.3
obs_model = ExpUniGauss(n_hidden_states)

obs_model.set_up_initials(priors=obs_model_init)

the_hmm = hmm_engine(obs_model, n_hidden_states)

priors = {'tpm': tpm, 'pi': startprob}
priors.update(obs_model_init)

the_hmm.set_up_initials(priors=priors)
no_paths=200
sequence_length= 10000

data = performance_statistics(hmm=the_hmm).generate_observations(sequence_length=sequence_length, no_paths=no_paths)

data_dates_list= biz_days_list(no_paths)

for x in range(0,no_paths):
    df=pd.DataFrame(data[x],columns=['ReturnTradedPrice', 'Duration'])
    df['TradedTime']=timestamps_(sequence_length)
    date_file = target_dir +'/'+ data_dates_list[x]+'.csv'
    df.to_csv(date_file)