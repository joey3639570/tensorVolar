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

#indexes = np.load("select_freq.npy")
fft = np.load("fft_all.npy")
labels = np.load("labels.npy")

# select frequencies and make train test subset
indexes = [1] + list(range(2,12001,4))
X = np.log(fft[:,indexes])
#print(X.shape)
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, labels, test_size=1/6, stratify=labels, random_state=103)
std = np.std(train_x, axis=0)
mean = np.mean(train_x, axis=0)
train_x = (train_x - mean)/std
test_x = (test_x - mean)/std

# grid search parameter
print("param searching...")
param_grid = {"C": 0.00001*np.power(2, range(10,24)),"gamma":0.0000001*np.power(2, range(2,12))}
grid_search = model_selection.GridSearchCV(svm.SVC(kernel="rbf"), param_grid, cv=10, return_train_score=False, verbose=2, n_jobs=32)
grid_search.fit(train_x, train_y)
best_params = grid_search.best_params_
print("best params: ", best_params)
print("best score: ", grid_search.best_score_)
print("best_estimator on test set:", grid_search.best_estimator_.score(test_x, test_y))

KFold = sklearn.model_selection.StratifiedKFold(n_splits=10)
all_svm = []
val_scores = []
for train_idx, val_idx in KFold.split(train_x, train_y): 
    svc = svm.SVC(C=best_params['C'], kernel="rbf", gamma=best_params['gamma'],verbose=False)
    svc.fit(train_x[train_idx], train_y[train_idx])
    val_scores.append(svc.score(train_x[val_idx], train_y[val_idx]))
    all_svm.append(svc)

print("average cross validation score: ", np.mean(val_scores))
predicts = [svc.predict(test_x) for svc in all_svm]
predicts = scipy.stats.mode(predicts, axis=0)[0]
accuracy = np.equal(predicts, test_y).astype(np.int32)
accuracy = np.mean(accuracy)
print(accuracy)
