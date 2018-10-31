"""
This file reads data from original data file from projectA
and performs fft transform on data, then use the magnitudes of different frequencies
as attribute to train SVM 
"""

import pandas as pd
import numpy as np
import scipy
from scipy import signal
import sklearn
from sklearn import svm
from sklearn import model_selection
import os

TRAIN_ROOT = "/projectA/920A"

def resample(array, num):
    indexes = np.linspace(0, len(array)-1, num, dtype=np.int32)
    return array[indexes]

def read_all_from_root(root):
    """
    This function reads data of projectA from the root of three types
    """
    data = []
    labels = []
    for type_dir in os.listdir(root):
        type_num = np.int32(type_dir[-1])
        print("reading data: type{}".format(type_num))
        for f in os.listdir(os.path.join(root,type_dir)):
            f = np.reshape(np.array(pd.read_csv(os.path.join(root,type_dir,f), header=None)), [-1])
            if len(f) < 512000:
                print(type_dir,f)
            data.append(resample(f, 512000))
            #data.append(signal.resample_poly(file, 512000))
            labels.append(type_num)
    return np.array(data), np.array(labels)



# read data
data, labels = read_all_from_root(TRAIN_ROOT)
data1027, label027 = read_all_from_root('/projectA/1027A')
#data = np.concatenate([data, data1027], axis=0)
#labels = np.concatenate([labels, label027], axis=0)
# fft transform
fft = np.fft.rfft(data)
del data
del data1027
# angle = np.angle(fft)
fft = np.abs(fft)
np.save("saved/labels.npy", labels)
np.save("saved/fft_all.npy", fft)
print("fft and labels saved.")
exit()

# select frequencies and make train test subset
indexes = [1] + list(range(2,4001,2))
X = np.log(fft[:,indexes])
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, labels, test_size=1/6, stratify=labels, random_state=101)
std = np.std(train_x, axis=0)
mean = np.mean(train_x, axis=0)
train_x = (train_x - mean)/std
test_x = (test_x - mean)/std

KFold = sklearn.model_selection.StratifiedKFold(n_splits=10)
all_svm = []
val_scores = []
for train_idx, val_idx in KFold.split(train_x, train_y):
    svc = svm.SVC(C=2.3, kernel="rbf", gamma=0.0005, verbose=False)
    svc.fit(train_x[train_idx], train_y[train_idx])
    val_scores.append(svc.score(train_x[val_idx], train_y[val_idx]))
    all_svm.append(svc)

print(val_scores)
predicts = [svc.predict(test_x) for svc in all_svm]
predicts = scipy.stats.mode(predicts, axis=0)[0]
accuracy = np.equal(predicts, test_y).astype(np.int32)
accuracy = np.mean(accuracy)
print(accuracy)
