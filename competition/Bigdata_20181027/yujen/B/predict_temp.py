from yen.projectB.readDatasets2 import Datasets
import numpy as np
import sklearn
from sklearn import svm

datasets = {}
for name in ["G14", "G35", "G57"]:
    datasets[name] = Datasets(name)

# load test data here

# test_data

# ===== calculate averge hold temp from training set ===== #
hold_dict ={}
for name, dataset in datasets.items():
    hold_temps = []
    for sample in dataset.samples:
        hold_temps += list(np.mean(np.sort(sample.getGoodData(),axis=1)[:,-5:], axis=1))
    hold_dict[name] = np.mean(hold_temps)
#    print('std: ', name, np.std(hold_temps))
 #   print('min: ', name, np.min(hold_temps))
  #  print('max: ', name, np.max(hold_temps))
print("hold times for each type: ", hold_dict)
exit()

# ===== SVM classification for types using 10 first temperature ===== #

X = []
labels = []
sample_len = 15
for name, dataset in datasets.items():
    for sample in dataset.samples:
        feature = np.mean(sample.getGoodData()[:,:sample_len], axis=0)
        X.append(feature)
        labels.append(name)

svc = svm.SVC(C=2)
svc.fit(X, labels)
predict_labels = svc.predict(test_data)
predict_temp = map(lambda x: hold_dict[x], predict_labels)
print(predict_temp)
