from stratagemdataprocessing.dbutils.mongo import MongoPersister
from stratagemdataprocessing.crypto.market.arctic.arctic_storage import ArcticStorage
import pytz
import os
import datetime as dt
import pandas as pd

# end_date = dt.datetime(2018, 6, 17, tzinfo=pytz.UTC)
# ticker = 'BTCUSD.PERP.BMEX'
# lob = storage.load_lob(ticker, start_date, end_date, as_df=True)
#
#print storage.load_trades(ticker, start_date, end_date)
# print storage.count_trades(ticker, start_date)

# Grab all the trades
local_cache = os.path.join(os.path.expanduser('~'), '.arcticcache')

def get_arctic_data(lob_or_trades='trades'):
    mongo_client = MongoPersister.init_from_config('arctic_crypto', auto_connect=True)
    storage = ArcticStorage(mongo_client.client)
    if lob_or_trades == 'trades':
        subdir = os.path.join(local_cache, 'Trades')
        download_call_back = storage.load_trades
        data_ranges = storage.trades_range()
    elif lob_or_trades=='lob':
        subdir = os.path.join(local_cache, 'LOB')
        download_call_back = storage.load_lob
        data_ranges = storage.lob_range()
    else:
        raise ValueError("You shouild be requesting one of LOB or trades")
    for symbol, (sd, ed) in data_ranges.iteritems():
        symbol_dir = os.path.join(subdir, symbol)
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)

        print "Reading data for {}".format(symbol)

        for dd in pd.date_range(sd, ed, freq='D'):
            data = download_call_back(symbol, sd, ed, as_df=True)
            if len(data) > 0:
                print "storing {}".format( dd.strftime('%Y%m%d'))
                data.to_csv(os.path.join(subdir, dd.strftime('%Y%m%d')))


if __name__ == '__main__':

    get_arctic_data()
    #get_arctic_data()
