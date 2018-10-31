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
# grid search parameter
#print("param searching...")
#param_grid = {"C": 0.00001*np.power(2, range(10,24)),"gamma":0.0000001*np.power(2, range(2,12))}
##param_grid = {'C': np.arange(2.5,10.6, 0.5), 'gamma': np.arange(0.0000256, 0.0001025, 0.000005)}
#grid_search = model_selection.GridSearchCV(svm.SVC(kernel="rbf"), param_grid, cv=10, return_train_score=False, verbose=2, n_jobs=32)
#grid_search.fit(train_x, train_y)
#best_params = grid_search.best_params_
#print("best params: ", best_params)
#print("best score: ", grid_search.best_score_)

best_params = {'C': 5.24288, 'gamma': 0.0002048}

KFold = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True)
all_svm = []
val_scores = []
for train_idx, val_idx in KFold.split(train_x, train_y): 
    svc = svm.SVC(C=best_params['C'], kernel="rbf", gamma=best_params['gamma'],verbose=False)
    svc.fit(train_x[train_idx], train_y[train_idx])
    val_scores.append(svc.score(train_x[val_idx], train_y[val_idx]))
    all_svm.append(svc)

print(val_scores)
print("average cross validation score: ", np.mean(val_scores))
