from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as pl
import sys
import pickle
import pandas as pd

import time
from sklearn.svm import SVC


mkl_path='/home/ak/Gitreps/MKLpy/' #path where the MKL module is saved
sys.path.append(mkl_path)
from MKLpy.metrics.pairwise import HPK_kernel
from MKLpy.algorithms import EasyMKL,RMKL,AverageMKL
from MKLpy.algorithms import KOMD
from MKLpy.regularization import normalization,rescale_01
from MKLpy.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import rbf_kernel as RBF



###main code starts here###
start_time=time.time()
##to be replaced with proper data###
X = np.random.randn(3000, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    # y = X[:,0]>0

y = np.array(y, dtype=np.int)
y[y == 0] = -1

######
K = HPK_kernel(X,degree=2)
KL = [HPK_kernel(X,degree=d) for d in range(1,11)] #kernel list
KL2 = [RBF(X, gamma=gamma) for gamma in [1., 10, 100.]]
X = rescale_01(X)
X = normalization(X)
#learn kernels
K_easy = EasyMKL(lam=0.1).arrange_kernel(KL2,y)
#fit models
clf_komd = KOMD(lam=0.1,kernel='precomputed').fit(K_easy,y)
clf_easy = EasyMKL().fit(KL,y)
clf_average = AverageMKL().fit(KL2,y)
# clf_svc  = SVC(C=10,kernel='rbf').fit(K_rmgd,Y)

KLtr,KLte,Ytr,Yte = train_test_split(KL2,y,train_size=.75,random_state=42)
y_score = clf_easy.fit(KLtr,Ytr).decision_function(KLte)


auc_score = roc_auc_score(Yte, y_score)
print("auc_score", auc_score)
print("--- %s seconds ---" % (time.time() - start_time))


EasyMKL().fit(KL2,Ytr)
