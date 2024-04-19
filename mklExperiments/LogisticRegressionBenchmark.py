import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, accuracy_score
import concurrent.futures

class CreateMarketFeatures:
    def __init__(self, df):
        self.df = df

    def ma_spread(self, short_window=5, long_window=10):
        self.df['ma_spread'] = self.df['TradedPrice'].rolling(window=long_window, min_periods=1).mean() - \
                               self.df['TradedPrice'].rolling(window=short_window, min_periods=1).mean()
        return self.df

    def obv_calc(self):
        volume_direction = self.df['Volume'] * np.sign(self.df['TradedPrice'].diff())
        self.df['OBV'] = volume_direction.cumsum()
        return self.df

    def labels(self):
        # Ensure label columns are correctly identified
        label_columns = [col for col in self.df.columns if 'label' in col]
        if not label_columns:
            raise ValueError("No label column found.")
        self.df['Label'] = self.df[label_columns[0]]  # Use the first label column found
        return self.df

    def chaikin_mf(self, period=5):
        high = self.df['TradedPrice'].rolling(window=period, min_periods=1).max()
        low = self.df['TradedPrice'].rolling(window=period, min_periods=1).min()
        close = self.df['TradedPrice']
        cmf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        self.df['CMF'] = cmf_multiplier * self.df['Volume'].rolling(window=period, min_periods=1).sum()
        self.df['CMF'].fillna(0, inplace=True)
        return self.df

def get_paths_for_symbol(symbol, FeaturesDir, LabelOne):
    features_path = os.path.join(FeaturesDir, symbol, 'MODEL_BASED')
    labels_path = os.path.join(LabelOne, symbol)
    return {'Features Path': features_path, 'Labels Path': labels_path, 'Symbol': symbol}

def load_and_process_data(symbol_paths):
    # Check if label path exists to avoid FileNotFoundError
    if not os.path.exists(symbol_paths['Labels Path']):
        print(f"Directory not found for symbol: {symbol_paths['Symbol']}")
        return pd.DataFrame()

    dates = sorted([f[:-4] for f in os.listdir(symbol_paths['Labels Path']) if f.endswith('.csv')])
    data_frames = []
    for date in dates:
        labels_file = os.path.join(symbol_paths['Labels Path'], date + '.csv')
        try:
            df = pd.read_csv(labels_file)
            market_features = CreateMarketFeatures(df)
            df = market_features.ma_spread()
            df = market_features.obv_calc()
            df = market_features.chaikin_mf()
            df = market_features.labels()
            df['Symbol'] = symbol_paths['Symbol']
            df['Date'] = date
            data_frames.append(df)
        except FileNotFoundError:
            print(f"File not found: {labels_file}")
            continue
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

FeaturesDir = '/media/ak/DataOnly1/SymbolFeatureDirectories/'
LabelOne = '/media/ak/DataOnly1/ExperimentCommonLocs/LabelsAlternateOne/'
symbols = sorted(os.listdir(FeaturesDir))
    #['AAL.L', 'APF.L','AV.L','AZN.L','BARC.L','BATS.L']
#

# Process data in parallel
all_results = pd.DataFrame()
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(load_and_process_data, get_paths_for_symbol(symbol, FeaturesDir, LabelOne)) for symbol in symbols]
    for future in concurrent.futures.as_completed(futures):
        result_df = future.result()
        if not result_df.empty:
            all_results = pd.concat([all_results, result_df], ignore_index=True)

# Assuming all_results is your final DataFrame ready for training the model
if not all_results.empty:
    # Fill NaN values. Here, using median as an example:
    X = all_results[['ma_spread', 'OBV', 'CMF']].fillna(all_results.median())
    y = all_results['Label'].fillna(method='ffill')  # Forward fill or use another method as appropriate

    # Check and handle any infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)  # Replace any new NaNs created by infinities

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    # Initialize and fit the logistic regression model
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
else:
    print("No data to process.")

