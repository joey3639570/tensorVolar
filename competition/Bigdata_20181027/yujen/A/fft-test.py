"""
This file reads data from original data file from projectA
and performs fft transform on data, then use the magnitudes of different frequencies
as attribute to train SVM 
"""

import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn import svm
from sklearn import model_selection
import os

TEST_ROOT = "/projectA/1027A_test"

def resample(array, num):
    indexes = np.linspace(0, len(array)-1, num, dtype=np.int32)
    return array[indexes]

assert os.path.isdir(TEST_ROOT), "TEST_ROOT is not valid"
data = []
for f in os.listdir(TEST_ROOT):
    f = np.array(pd.read_csv(os.path.join(TEST_ROOT,f))).reshape([-1])
    data.append(resample(f, 512000))

# fft transform
fft = np.fft.rfft(data)
del data
# angle = np.angle(fft)
fft = np.abs(fft)
np.save("saved/fft_test.npy", fft)
np.save("saved/test_files.npy", np.array(os.listdir(TEST_ROOT)))
print("fft_test.npy saved.")
exit()

