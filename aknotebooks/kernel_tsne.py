
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
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

RS = 123

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

# Utility function to visualize the outputs of PCA and t-SNE

def fashion_scatter(x, colors, locationFileName):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    plt.rcParams["figure.autolayout"] = True
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 8))
    lim = (x.min()-5, x.max()+5)
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["left"].set_visible(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_visible(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax1.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)], alpha =0.2)
    ax2.scatter(x[:,0], x[:,2], lw=0, s=40, c=palette[colors.astype(np.int)], alpha =0.2)

    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)

    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)
    ax2.xaxis.set_tick_params(labelsize=10)
    ax2.yaxis.set_tick_params(labelsize=10)
    ax2.set_xlabel('t-SNE 1', fontsize = 13)
    ax1.set_ylabel('t-SNE 2', fontsize = 13)
    ax2.set_ylabel('t-SNE 3', fontsize = 13)

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    ax1.axis('tight')
    ax2.axis('tight')
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])
    ax2.autoscale() ## call autoscale if needed
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(locationFileName)

    plt.show()



    return fig, ax1, ax2


if __name__ == '__main__':

    symbolIdx = 1

    symbolSpecificFeaturesDirectors = os.path.join(symbolFeaturesDirectories, sorted(symbols)[symbolIdx])
    print('you chose symbol:', sorted(symbols)[symbolIdx])
    symbolFeaturesLocation = "/".join((symbolSpecificFeaturesDirectors, 'MODEL_BASED'))  # where all the HMM output is
    print('your features are here-in pickle format:',symbolFeaturesLocation)
    # and a label Idx

    time_start = time.time()


    for labelsIdx in ['One', 'Two', 'Three', 'Four']:
        print('you are picking Label', labelsIdx)
        LabelsFilesPath = os.path.join(labelsLocations[str(labelsIdx)], sorted(symbols)[symbolIdx])
        LabelsFiles = os.listdir(LabelsFilesPath)

        dates = os.listdir(symbolFeaturesLocation)

        for dateIdx in [7]:

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

            X_array = np.asarray(X)
            is_all_zero = np.all(X_array == 0)
            if is_all_zero:
                print('array is all zeros')
            else:
                print('Array is good')
                choice_length= np.count_nonzero(~np.isnan(labels))

                X, y = shuffle(X_array, labels)
                X = X[:choice_length]
                y = y[:choice_length].fillna(0)

                scaler = MinMaxScaler(feature_range=(-1, 1))
                mm = make_pipeline(MinMaxScaler(), Normalizer())
                X = mm.fit_transform(X)
                rbf_feature = RBFSampler(gamma=1.5, random_state=10)
                ps = PolynomialCountSketch(degree=11, random_state=1)
                X_rbf_features = rbf_feature.fit_transform(X)
                X_poly_features = ps.fit_transform(X)
                # We want to get TSNE embedding with 2 dimensions
                n_components = 3
                tsne = TSNE(n_components)
                tsne_result = tsne.fit_transform(X_rbf_features)
                locationFileName = os.path.join(figuresDestination,str(sorted(symbols)[symbolIdx])+'_idx_'+
                                                str(idx)+
                                                'date_'+str(dateIdx)+'_'+str(labelName)+'_tsne_rbf_kernelised.png')

                fashion_scatter(tsne_result, y, locationFileName)


                fig = plt.figure(figsize=(16,9))
                ax =plt.axes(projection = '3d')
                # ax = Axes3D(fig)
                df =pd.DataFrame(tsne_result)
                x =df[0]
                y = df[1]
                z = df[2]

                df = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=range(len(x)))
                trisurf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.3, antialiased = True, edgecolor = 'grey')
                ax.zaxis.set_tick_params(labelsize=10)
                ax.xaxis.set_tick_params(labelsize=10)
                ax.yaxis.set_tick_params(labelsize=10)
                # Adding labels
                ax.set_xlabel('t-sne 1', fontsize=13, rotation = 90)
                ax.set_ylabel('t-sne 2', fontsize=13, rotation = -90)
                ax.set_zlabel('t-sne 3', fontsize=13, rotation = -90)

                ax.view_init(-140, 100)
                df = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=range(len(x)))
                trisurf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.3, antialiased = True, edgecolor = 'grey')
                ax.zaxis.set_tick_params(labelsize=10)
                ax.xaxis.set_tick_params(labelsize=10)
                ax.yaxis.set_tick_params(labelsize=10)
                # Adding labels
                ax.set_xlabel('t-sne 1', fontsize=13, rotation = 90)
                ax.set_ylabel('t-sne 2', fontsize=13, rotation = -90)
                ax.set_zlabel('t-sne 3', fontsize=13, rotation = -90)

                ax.view_init(-140, 100)
                locationFileName_3D = os.path.join(figuresDestination,str(sorted(symbols)[symbolIdx])+'_idx_'+
                                                   str(idx)+
                                                   'date_'+str(dateIdx)+'_'+str(labelName)+'_tsne_rbf_kernelised_3D.png')
                plt.savefig(locationFileName_3D)

                plt.show()
