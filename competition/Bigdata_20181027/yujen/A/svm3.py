"""
This file reads data from presaved numpy array that are already fft transformed
and uses selected frequencies as attributes to train SVM 
"""

import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn import svm
from sklearn import model_selection
import os

indexes = np.load("saved/select_freq.npy")
fft = np.load("saved/fft_all.npy")
labels = np.load("saved/labels.npy")

# select frequencies and make train test subset
# indexes = [1] + list(range(2,12001,4))
X = np.log(fft[:,indexes])
std = np.std(X, axis=0)
mean = np.mean(X, axis=0)
train_x = (X - mean)/std
train_y = labels

best_params = {'C': 3.5, 'gamma': 7.56e-05}

KFold = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=102)
all_svm = []
val_scores = []
for train_idx, val_idx in KFold.split(train_x, train_y): 
    predicts = []
    for i in range(3):
        svc = svm.SVC(C=best_params['C'], kernel="rbf", gamma=best_params['gamma'],verbose=False)
        svc.fit(train_x[train_idx,i::3], train_y[train_idx])
        predicts.append(svc.predict(train_x[val_idx,i::3]))
        all_svm.append(svc)
    predict = scipy.stats.mode(predicts, axis=0)[0]
    print(predict)
    val_scores.append(sklearn.metrics.accuracy_score(train_y[val_idx], predict[0]))

print(val_scores)
print("average cross validation score: ", np.mean(val_scores))
