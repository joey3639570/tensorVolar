import os
import numpy as np
import pandas as pd
import tools

class Dataset:
    def __init__(self, LengthOfSeq, NumOfVars):
        self.dataList = np.arange(LengthOfSeq*NumOfVars).reshape(LengthOfSeq, NumOfVars)
        self.quality = 0

    def readFile(self, fullFilename):
        self.dataFile = pd.read_excel(io=fullFilename, header=None)
        self.quality = np.float32(self.dataFile.iloc[-1, 0].split(':')[-1])
        self.dataList = np.array(self.dataFile.iloc[0:-1], dtype=np.float32)
        #self.dataList = tools.zscore(self.dataList)
        #self.dataList = np.log(self.dataList)
        #self.dataList = (self.dataList - np.mean(self.dataList))/np.std(self.dataList)
        #self.quality = np.log(self.quality)

def isXlsFile(filename, file_dir):
    """Check if the filename contain a 'xls' ext at the end"""
    filename = os.path.join(file_dir, filename)
    [name, ext] = os.path.splitext(filename)
    if os.path.isfile(filename):
        if ext == '.xls':
            return 1
        else:
            return 0

def getXlsFile(data_dir):
    """Get all filename in the specified directory"""
    fullFilename = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if isXlsFile(f, data_dir) == 1]
    return fullFilename

def sortByQuality(dataset):
    """The sorting rule for the function 'getSortedDataset'"""
    return dataset.quality

def getSortedDataset(data_dir, LengthOfSeq, NumOfVars):
    """Return a list of sorted datasets"""
    xlsFileList = getXlsFile(data_dir)
    datasets = []
    for filename in xlsFileList:
        newDataset = Dataset(LengthOfSeq, NumOfVars)
        newDataset.readFile(filename)
        datasets.append(newDataset)
    datasets.sort(key=sortByQuality, reverse=False)
    return datasets
