from yen.projectB.readDatasets2 import Datasets
import numpy as np
import sklearn
from sklearn import svm

Datasets.set_root('/projectB/920B')
datasets = {}
for name in ["G14", "G35", "G57"]:
    datasets[name] = Datasets(name)

# ===== SVM classification for types using 10 first temperature ===== #

X = []
labels = []
sample_len = 15
for name, dataset in datasets.items():
    for sample in dataset.samples:
        averaged_first_ten = np.mean(sample.getGoodData()[:,:sample_len], axis=0)
        X.append(averaged_first_ten)
        labels.append(name)

train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X, labels, stratify=labels, test_size=0.2)
svc = svm.SVC(C=2)
svc.fit(train_x, train_y)
