from os import sys, path
#from convenience_functions.classification_utilities import *
from time import time
from sklearn.model_selection import train_test_split
from scipy import sparse, io
from sklearn.decomposition import PCA
import sklearn.preprocessing as prep

sc= prep.StandardScaler()##replace this with other scalers to investigate performance
min_max_scaler = prep.MinMaxScaler()
max_abs_scaler = prep.MaxAbsScaler()
robust_scaler = prep.RobustScaler() #robust against outliers

def split_data(X, y,tsize):
    training_data, test_data, training_labels, test_labels = train_test_split(
        X, y, test_size=tsize, random_state=0)
    return training_data, test_data, training_labels, test_labels


def normalised_split_data(X, y,tsize, sclr):

    training_data, test_data, training_labels, test_labels = split_data(X, y,tsize)
    training_data_transformed = sclr.fit_transform(training_data)
    test_data_transformed = sclr.fit_transform(test_data)

    return training_data_transformed, test_data_transformed, training_labels, test_labels

def dimensionality_reduction(training_data, test_data,n_compo, type='pca'):
    if type == 'pca':
        n_components = n_compo
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        pca.fit(training_data)
        print("done in %0.3fs" % (time() - t0))
        t0 = time()
        training_data_transform = sparse.csr_matrix(pca.transform(training_data))
        test_data_transform = sparse.csr_matrix(pca.transform(test_data))
        print("done in %0.3fs" % (time() - t0))
        #random_projections
        #feature_agglomeration
        return training_data_transform, test_data_transform

def split_data(X, y,tsize):
    training_data, test_data, training_labels, test_labels = train_test_split(
        X, y, test_size=tsize, random_state=0)
    return training_data, test_data, training_labels, test_labels


def normalised_split_data(X, y,tsize, sclr):

    training_data, test_data, training_labels, test_labels = split_data(X, y,tsize)
    training_data_transformed = sclr.fit_transform(training_data)
    test_data_transformed = sclr.fit_transform(test_data)

    return training_data_transformed, test_data_transformed, training_labels, test_labels

def dimensionality_reduction(training_data, test_data,n_compo, type='pca'):
    if type == 'pca':
        n_components = n_compo
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        pca.fit(training_data)
        print("done in %0.3fs" % (time() - t0))
        t0 = time()
        training_data_transform = sparse.csr_matrix(pca.transform(training_data))
        test_data_transform = sparse.csr_matrix(pca.transform(test_data))
        print("done in %0.3fs" % (time() - t0))
        #random_projections
        #feature_agglomeration
        return training_data_transform, test_data_transform

    def normalise_df(df):
        df_norm = pd.DataFrame()
        detominator = df.index.values + 1
        for col in list(df.columns):
            df_norm[col] = df[col] / (denominator)
        return df_norm

# def readFile(path_, fname_):
#     '''Reads in a file for dataset.
#
#     Args:
#         fname: a string, the file name
#
#         pred: the file we are reading has prediction. Default = False.
#
#     Returns:
#         xs: a list of list of feature values
#         ys: a list of predictions
#     '''
#     sys.path.append(path_)
#     pattern_ ='features*.csv'
#     list_features = glob.glob(pattern_) #makes a list of all the feature files
#     scaled_features_list =[]
#     for filename in list_features:
