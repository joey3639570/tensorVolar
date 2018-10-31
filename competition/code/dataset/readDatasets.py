import os
import numpy as np
import pandas as pd
#import sys
#import codecs
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Directory of datasets
JobDir = '/home/t125501/workplace/csvprojectB/'

# Dataset name (Each dataset need to have indiviual directory named with this list
# , and the index of this list called 'data_group'
datasetName = ['G14', 'G35', 'G57']

# The position of data in csv file, differ from datasets
dataset_content = np.array([[18, 93], [16, 24], [16, 40]])

# Definition of bad data
BAD_DATA_CRITERIA = 400

def isCSVFile(filename, file_dir):
    filename = os.path.join(file_dir, filename)
    [name, ext] = os.path.splitext(filename)
    if os.path.isfile(filename):
        if ext == '.csv':
            return 1
        else:
            return 0

def getCSVFile(data_dir):
    fullFilename = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if isCSVFile(f, data_dir) == 1]
    return fullFilename

class Datasets:
    '''
    A Datasets object contain many samples, which store in the data member, 'datasets'.
    '''
    def __init__(self, dataset_name):
        assert(dataset_name in datasetName), "Dataset must be one of G14, G35, G57"
        index = datasetName.index(dataset_name)
        dataset_path = os.path.join(JobDir, dataset_name)
        csvFileList = getCSVFile(dataset_path)
        self.samples = []
        for filename in csvFileList:
            self.samples.append(Dataset(filename, index))

class Dataset:
    '''
    Data member introduction:
    dataFile -> Original csv file
    dataList -> 2d Array of sensor value
    dataTime -> 1d Array of data time label (Time)
    dataLabel -> 1d Array of sensor name    (T1~TX)
    goodData -> 1d boolean Array
    startTime -> Integer of start time (In minute)
    '''
    def __init__(self, data_filename, data_group):
        '''
        data_filename: String of single csv filename
        data_group: The ID of the dataset, that is, the index of dataset in 'datasetName'
        '''
        # Read dataset
        self.dataFile = pd.read_csv(data_filename)
        self.dataList = self.dataFile.iloc[5:, dataset_content[data_group, 0]:dataset_content[data_group, 1]]
        self.dataTime = self.dataFile.iloc[5:,2]
        self.dataLabel = self.dataFile.iloc[3, dataset_content[data_group, 0]:dataset_content[data_group, 1]]
        # Transpose
        self.dataList = np.transpose(np.array(self.dataList)).astype(np.float)
        self.dataTime = np.transpose(np.array(self.dataTime))
        startTimeInfo =  self.dataTime[0].split(':')
        self.startTime = startTimeInfo[1]
        self.goodData = np.ones(dataset_content[data_group, 1] - dataset_content[data_group, 0] + 1, dtype=bool)
        self.checkBadData()

    def isBadData(self, dataList):
        '''
        Check if a single sensor data list is a bad data
        Definition of bad data: Any data exceed 'BAD_DATA_CRITERIA'
        '''
        for data in dataList:
            if(data >= BAD_DATA_CRITERIA):
                return True
        return False

    def checkBadData(self):
        '''
        Delete any bad data
        '''
        for index in range(len(self.dataList)):
            if(self.isBadData(self.dataList[index])):
                self.dataLabel[index] = 'BAD'
                self.goodData[index] = False

class MiniDataset:
    def __init__(self, datasets, size=10, decideIndex=15):
        '''
        Data member introduction:
        data -> 2d Array of sensor value, shape=[sample, size, length]
        startTime -> 1d Array of each sample start time, shape=[sample]
        '''
        self.miniSize = size
        numOfSample = len(datasets.samples)
        numOfRecord = np.min([sample.dataList.shape[1] for sample in datasets.samples])
        numOfSensor = datasets.samples[0].dataList.shape[0]
        self.data = np.ones((numOfSample, size, numOfRecord), dtype=np.float32)
        self.startTime = np.ones(numOfSample, dtype=int)
        self.lastSensorDecideTimeIndex = decideIndex
        self.printInfo = False
        # Find last sensors
        for sampleIndex in range(numOfSample):
            sensors = [[index, datasets.samples[sampleIndex].dataList[index, self.lastSensorDecideTimeIndex]] for index in range(numOfSensor)]
            sensors = sorted(sensors, key=lambda s: s[1])
            arrayIndex = 0
            if(self.printInfo):
                print(sorted([sensors[i][0] for i in range(self.miniSize)]))
                print("Choosed index in sample %d: " %sampleIndex, end=' ')
            for index in range(self.miniSize):
                if(self.printInfo):
                    print(sensors[index][0], end=' ')
                self.data[sampleIndex, index] = datasets.samples[sampleIndex].dataList[ sensors[index][0] , 0:numOfRecord]
                self.startTime[arrayIndex] = datasets.samples[sampleIndex].startTime
            if(self.printInfo):
                print('')

def main():
    print("\n\n")
    print("       ===========================================")
    print("       =========User Guild of readDataset=========")
    print("       ===========================================")
    print("Class Datasets Example:")
    print("Declare a Dataset object:        g14 = Datasets('G14')")
    print("Using first sample dataList:     g14.samples[0].dataList")
    print("Obtain first sample start time:  g14.samples[0].startTime")
    print("")

    print("Class MiniDataset Example")
    print("Choose 10 lowest temperature sensor at 15th time point")
    print("Declare a MiniDatasetobject:     miniG14 = MiniDataset(g14, 10, 15)")
    print("Using first sample dataList:     miniG14.data[0]")
    print("Using first sample startTime:    miniG14.startTime[0]")

    print("\nMore info please check the comment in this program\n\n")

    g14 = Datasets('G14')
    #g35 = Datasets('G35')
    #g57 = Datasets('G57')

    print("Number of g14 sample[0] sensors:   " + str(len(g14.samples[0].dataList)))
    print("Start time of sample[0] (minute):  " + str(g14.samples[0].startTime))
    miniG14 = MiniDataset(g14)
    print("MiniG14 of size 10: ")
    print(miniG14.data[0])

if __name__ == "__main__":
    main()