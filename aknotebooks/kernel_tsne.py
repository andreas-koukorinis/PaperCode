
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder

from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.kernel_approximation import RBFSampler

import seaborn as sns

import pandas as pd
import numpy as np
import sklearn.svm as svm
import pickle
import os

from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
standard_scaler = StandardScaler()
import pickle as pkl
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder
# Feature Analysis Imports

from sklearn.cluster import KMeans

### matplotlib stuff ##

from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl
import matplotlib.pyplot as plt
styleFiles = os.listdir(mpl.get_configdir())
styleFileIdx = 2
print(styleFiles[styleFileIdx])
plt.style.use(os.path.join(mpl.get_configdir(), styleFiles[styleFileIdx]))
from sklearn.manifold import TSNE
# locations ##
figuresDestination ='/home/ak/Documents/Research/Papers/figures/'
hardDriveLoc = '/media/ak/'
dataOnlyFolder = '/media/ak/DataOnly'  # external date only drive
symbolFeaturesDirectories  = os.path.join(dataOnlyFolder, 'SymbolFeatureDirectories')
ExperimentCommonLocs = os.path.join(dataOnlyFolder,'ExperimentCommonLocs')

labelsLocations = {'One': os.path.join(ExperimentCommonLocs, 'LabelsAlternateOne'),
                   'Two': os.path.join(ExperimentCommonLocs, 'LabelsAlternateTwo'),
                   'Three': os.path.join(ExperimentCommonLocs, 'LabelsAlternateThree'),
                   'Four': os.path.join(ExperimentCommonLocs, 'LabelsAlternateFour'),
                   'File':os.path.join(ExperimentCommonLocs, 'LabelsAlternateFive'),
                   'Six': os.path.join(ExperimentCommonLocs, 'LabelsAlternateSix')}

# symbols

symbols = sorted(os.listdir(symbolFeaturesDirectories))


def open_pickle_file(path, pickle_file):
    file_loc = os.path.join(path, pickle_file)
    pickle_to_file = pkl.load(open(file_loc, "rb"), encoding='iso-8859-1')
    return pickle_to_file


if __name__ == '__main__':

    symbolIdx = 10


    symbolSpecificFeaturesDirectors = os.path.join(symbolFeaturesDirectories, sorted(symbols)[symbolIdx])
    print('you chose symbol:', sorted(symbols)[symbolIdx])
    symbolFeaturesLocation = "/".join((symbolSpecificFeaturesDirectors, 'MODEL_BASED'))  # where all the HMM output is
    print('your features are here-in pickle format:',symbolFeaturesLocation)
    # and a label Idx


    for labelsIdx in ['One', 'Two', 'Three', 'Four']:
        print('you are picking Label', labelsIdx)
        LabelsFilesPath = os.path.join(labelsLocations[str(labelsIdx)], sorted(symbols)[symbolIdx])
        LabelsFiles = os.listdir(LabelsFilesPath)

        dates = os.listdir(symbolFeaturesLocation)

        for dateIdx in [5, 7, 10, 14]:
            idx =4
            dateSpecificDirectory = os.path.join(symbolFeaturesLocation, dates[dateIdx])
            dateSpecificFiles = sorted(os.listdir(dateSpecificDirectory))
            print("dateIdx", dateIdx, "_idx_", idx)

            specificFile = os.path.join(dateSpecificDirectory, dateSpecificFiles[idx])

            featuresTupleFile  = open_pickle_file(dateSpecificDirectory, dateSpecificFiles[idx])

            label_date = specificFile.split("_")[6]

            labelDateFile = os.path.join(LabelsFilesPath, str(label_date) +'.csv')

            if os.path.isfile(labelDateFile):
                df = pd.read_csv(labelDateFile)
            else:
                print('not a valid dates file')

            labelName = str(df.columns[df.columns.str.contains(pat='label')].values[0])
            labels = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]

            dfFeatures = pd.concat([featuresTupleFile[0], featuresTupleFile[1], \
                                    featuresTupleFile[2], featuresTupleFile[3]], axis=1, sort=False).fillna(0)


            features= list(dfFeatures.columns.values)
            X = pd.get_dummies(dfFeatures)
            # Convert unique classes (strings) into integers

            X_array = np.asarray(X)
            print(X_array)

            X_array = np.asarray(X)
            is_all_zero = np.all(X_array == 0)
            if is_all_zero:
                print('array is all zeros')
            else:
                print('Array is good')
                print(X_array)


                choice_length= np.count_nonzero(~np.isnan(labels))

                X, y = shuffle(X_array, labels)
                X = X[:choice_length]
                y = y[:choice_length].fillna(0)

                scaler = MinMaxScaler(feature_range=(-1, 1))
                mm = make_pipeline(MinMaxScaler(), Normalizer())
                X_train = mm.fit_transform(X)
                rbf_feature = RBFSampler(gamma=1.5, random_state=10)
                ps = PolynomialCountSketch(degree=11, random_state=1)
                X_rbf_features = rbf_feature.fit_transform(X_train)
                X_poly_features = ps.fit_transform(X_train)
                # We want to get TSNE embedding with 2 dimensions
                n_components = 2
                tsne = TSNE(n_components)
                tsne_result = tsne.fit_transform(X_rbf_features)
                tsne_result.shape

                tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})

                fig, ax = plt.subplots(1,facecolor='White')

                x=tsne_result_df['tsne_1']
                y=tsne_result_df['tsne_2']
                plt.rcParams["figure.autolayout"] = True
                lim = (tsne_result.min()-5, tsne_result.max()+5)
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.spines["left"].set_visible(True)
                ax.set_aspect('equal')
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                plt.scatter(x, y,  alpha=0.2, s=50, cmap=plt.cm.rainbow)
                plt.xlabel('T-SNE 1', fontsize = 13)
                plt.ylabel('T-SNE 2', fontsize = 13)
                plt.axis("tight")
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.grid(True)
                locationFileName = os.path.join(figuresDestination,str(sorted(symbols)[symbolIdx])+'_idx_'+
                                                str(idx)+
                                                'date_'+str(dateIdx)+'_label'+str(labelName)+'_tsne_rbf_kernelised.png')
                plt.savefig(locationFileName)
                plt.tight_layout()
                plt.show()

                ## poly kernels

                # We want to get TSNE embedding with 2 dimensions
                n_components = 2
                tsne = TSNE(n_components)
                tsne_result = tsne.fit_transform(X_poly_features)
                tsne_result.shape
                # Plot the result of our TSNE with the label color coded
                # A lot of the stuff here is about making the plot look pretty and not TSNE

                tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})

                fig, ax = plt.subplots(1,facecolor='White')

                x=tsne_result_df['tsne_1']
                y=tsne_result_df['tsne_2']
                plt.rcParams["figure.autolayout"] = True
                lim = (tsne_result.min()-5, tsne_result.max()+5)
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.spines["left"].set_visible(True)
                ax.set_aspect('equal')
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                plt.scatter(x, y,  alpha=0.3, s=50, cmap=plt.cm.rainbow)
                plt.xlabel('T-SNE 1', fontsize = 13)
                plt.ylabel('T-SNE 2', fontsize = 13)

                plt.axis("tight")
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.grid(True)
                plt.grid(True)
                plt.legend()

                plt.grid(True)
                locationFileName = os.path.join(figuresDestination,str(sorted(symbols)[symbolIdx])+'_idx_'+
                                                str(idx)+
                                                'date_'+str(dateIdx)+'_label'+str(labelName)+'_tsne_poly_kernelised.png')
                plt.savefig(locationFileName)

