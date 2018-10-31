import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import *


import tensorflow as tf
import pywt

# Project Info
JobDir = '/home/cch/Downloads/joey'
TrainingDataDir = os.path.join(JobDir, '806traindata')
# Dataset Info
LengthOfSeq = 7500
NumOfVars = 4
NumOfData = 40

class Dataset:
    def __init__(self, fullFilename):
        self.dataList = np.arange(LengthOfSeq*NumOfVars).reshape(NumOfVars, LengthOfSeq)
        self.dataFile = pd.read_excel(io=fullFilename, header=None)
        self.quality = np.float32(self.dataFile.iloc[-1, 0].split(':')[-1])
        self.dataList = self.dataFile.iloc[0:-1]

def isXlsFile(filename, file_dir):
    filename = os.path.join(file_dir, filename)
    [name, ext] = os.path.splitext(filename)
    if os.path.isfile(filename):
        if ext == '.xls':
            return 1
        else:
            return 0

def getXlsFile(data_dir):
    fullFilename = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if isXlsFile(f, data_dir) == 1]
    return fullFilename

def sortByQuality(dataset):
    return dataset.quality

def getSortedDataset(data_dir):
    xlsFileList = getXlsFile(data_dir)
    datasets = []
    for filename in xlsFileList:
        datasets.append(Dataset(filename))
    datasets.sort(key=sortByQuality, reverse=False)
    return datasets

pd.options.display.max_rows = 10
datasets = getSortedDataset(TrainingDataDir)

#for i in datasets:
#    print("Dataset Quality= ", i.quality)
#    print(i.dataList)

def scalogram(data):
    bottom = 0

    vmin = min(map(lambda x: min(abs(x)), data))
    vmax = max(map(lambda x: max(abs(x)), data))

    gca().set_autoscale_on(False)

    for row in range(0, len(data)):
        scale = 2.0 ** (row - len(data))

        imshow(
            array([abs(data[row])]),
            interpolation = 'nearest',
            vmin = vmin,
            vmax = vmax,
            extent = [0, 1, bottom, bottom + scale])

        bottom += scale

def centerd_moving_average(array, size):
    sumed = np.zeros(len(array)-size)
    for i in range(size):
        sumed += array[i:-size+i]
    return sumed/size

"""
def data_preprocessing(custom_datasets):
    size = len(custom_datasets)
    moving_average_size = 500
    for i in range(size):
        for j in range(4):
            col = np.array(datasets[i].dataList[j],dtype=np.float32)
            datasets[i].dataList[j] = centerd_moving_average(col, moving_average_size)
data_preprocessing(datasets)
"""

col = np.array(datasets[0].dataList[0],dtype=np.float32)
plt.plot(col)
plt.show()
averaged = centerd_moving_average(col , 500)
plt.plot(averaged)
plt.show()
afterft = pywt.wavedec(averaged,'db2')
print(afterft[0])
plt.plot(afterft[0])
plt.show()

print(afterft[1])
plt.plot(afterft[1])
plt.show()
"""
plt.figure(figsize=[120,30])
for i in range(40):
    for j in range(4):
        averaged = centerd_moving_average(datasets[i].dataList[j] , 500)
        afterft = pywt.wavedec(averaged,'db2')
        plt.subplot(4,40,i*4+j+1)
        plt.plot(afterft[0])
plt.tight_layout()
plt.show()
"""
