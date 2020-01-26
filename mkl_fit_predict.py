import os
import pandas as pd
import numpy as np
import pickle as pkl
import aknotebooks.classification.convenience_functions.mkl_base as mkl_base
from aknotebooks.classification.convenience_functions.mkl_base import hardDrivesLoc, dataOnlyDrive, folderList, \
    dataList, finalLocation, DataLoc, path

np.seterr(divide='ignore', invalid='ignore')

MKLExpPath = os.path.join(DataLoc, path)
symbols = sorted(os.listdir(MKLExpPath))

if __name__ == '__main__':
    symbolIdx = 0
    print(symbols[symbolIdx])
    MKLSymbolPath = os.path.join(MKLExpPath, symbols[symbolIdx])
    MKLSymbolKernelsPath = "/".join((MKLSymbolPath, 'Kernels'))

    cleanListKernelInputKeys = pkl.load(open("/".join((MKLSymbolKernelsPath, "cleanKernelsList.pkl",)), "rb"),
                                        encoding='latin1')
    for kernelKey in cleanListKernelInputKeys:
        kernelFileName = "/".join((MKLSymbolKernelsPath, "".join((kernelKey[0], '_', kernelKey[1], "_RBFKernels.pkl"))))
        RBFKernels = pkl.load(open(kernelFileName, "rb"), encoding='latin1')[0]  # read the kernels
