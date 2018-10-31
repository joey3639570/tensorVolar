import os
import numpy as np
import pandas as pd
import datetime
import yen.projectB.ProjectB_Tools as tools
from time import mktime

# Dataset name (Each dataset need to have indiviual directory named with this list
# , and the index of this list called 'data_group'
datasetName = ['G14', 'G35', 'G57', 'G29', 'G30', 'G44']

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
    # Directory of datasets, set this using static method
    root = '/home/t125501/workplace/projectB/920B'
    
    @staticmethod
    def set_root(path):
        Datasets.root = path

    def __init__(self, dataset_name, test=False):
        #assert(dataset_name in datasetName), "Dataset must be one of G14, G35, G57"
        #if Datasets.root == '/projectB/1027B_test':
        if test:
            dataset_path = Datasets.root
        else:
            dataset_path = os.path.join(Datasets.root, dataset_name)

        index = datasetName.index(dataset_name)
        dataset_path = os.path.join(self.root, dataset_name)
        print('reading dataset from root directory: ', dataset_path)
        self.samples = []
        for filename in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, filename)
            self.samples.append(Dataset(file_path, index))
    def __add__(self, that):
        self.samples += that.samples
        return self

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
                # set start time
        if Datasets.root == '/projectB/1027B': 
            self.dataFile = pd.read_excel(data_filename)
            self.dataLabel = list(self.dataFile.keys())
            #self.startTime = datetime.datetime(self.dataFile.iloc[0,0])
            # self.startTime = datetime.datetime(*(self.dataFile.iloc[0,0])[:6])
            self.startTime = datetime.datetime.combine(datetime.date.today(), self.dataFile.iloc[0,0])
            self.dataFile = self.dataFile.filter(regex="^T[0-9]", axis=1)
        else:
            # Read dataset
            self.dataFile = pd.read_excel(data_filename)
            self.dataFile = self.dataFile.loc[:, self.dataFile.iloc[1] != 'X']
            self.dataFile.columns = self.dataFile.iloc[2]
            self.dataFile = self.dataFile.iloc[4:]
            self.dataTime = "{}-{}".format( self.dataFile.iloc[0,0], str(self.dataFile.iloc[0,1]))
            self.startTime = datetime.datetime.strptime(self.dataTime, "%d/%m/%Y-%H:%M:%S") 
            self.dataFile = self.dataFile.filter(regex="^T[0-9]", axis=1)
            self.dataLabel = list(self.dataFile.keys())

        
#        print(self.dataFile)
        # Transpose
        self.dataList = np.transpose(np.array(self.dataFile), [1,0])
        self.goodData = np.ones(self.dataList.shape[0], dtype=bool)
        self.checkBadData()
        #print(self.dataList)

    def isBadData(self, dataList):
        '''
        Check if a single sensor data list is a bad data
        Definition of bad data: Any data exceed 'BAD_DATA_CRITERIA'
        '''
        condition = np.logical_or(dataList >= BAD_DATA_CRITERIA, dataList < 0)
        if np.count_nonzero(condition) == 0:
            return False
        else:
            return True

    def checkBadData(self):
        '''
        Delete any bad data
        '''
        for index, sensor in enumerate(self.dataList):
            if(self.isBadData(sensor)):
                self.dataLabel[index] = 'BAD'
                self.goodData[index] = False
    
    def getGoodData(self):
        #print(self.dataList.shape)
        #print(self.goodData.shape)
        return self.dataList[self.goodData]

class MiniDataset:
    def __init__(self,typeInfo, datasets):
        '''
        This class will construct a object which offer the last 'size' data 
        in growing temperature. It will choose a specific index to compare
        (decideIndex), sort it,  and put into 'data' array.

        Parameters:
        datasets -> An object of 'Datasets'
        size -> Outout size
        decideIndex -> Choose which index to compare

        Data member introduction:
        data -> 2d Array of sensor value, shape=[sample, size, length]
        startTime -> 1d Array of each sample start time, shape=[sample]
        ''' 
        self.printInfo = False

        self.startTime = []
        self.endTime = []
        numSensors = []
        numRecords = []
        for sample in datasets.samples:
            self.startTime.append(sample.startTime)
            numSensors.append(sample.getGoodData().shape[0])
            numRecords.append(sample.dataList.shape[1])
        # print(numRecords)
        numSensors = np.min(numSensors)
        numRecords = np.min(numRecords)

        if self.printInfo:
            print('number of selected sensors in miniDataset: ', numSensors)
            print('number of kept records in miniDatasets: ', numRecords)
        
        self.data = []
        self.endMaxTemp = []
        self.endMinTemp = []
        for sample_idx, sample in enumerate(datasets.samples):
            dataList = sample.getGoodData()
            sort_idx = np.argsort(dataList[:,typeInfo.LAST_DATA_DECIDE_INDEX])
            sorted_dataList = dataList[sort_idx]
            self.data.append(sorted_dataList[:numSensors, :numRecords])
            convIndex = tools.getConvIndex(typeInfo, sorted_dataList[0])
            et = sample.startTime + datetime.timedelta(seconds=int(convIndex * 60))
            self.endTime.append(et)
            self.endMaxTemp.append(np.max(dataList[:, convIndex]))
            self.endMinTemp.append(np.min(dataList[:, convIndex]))

        self.data = np.array(self.data)
        self.startTime = np.array(self.startTime)
        self.endTime = np.array(self.endTime)
        self.endMaxTemp = np.array(self.endMaxTemp)
        self.endMinTemp = np.array(self.endMinTemp)
        self.miniSize = numSensors

    def getTrainingData(self, dataLen, trigger):
        '''
        Slice the original datasets into pieces and put into a same array.
        # Parameter:
        dataLen -> Length of each training data.
        trigger -> The definition of 'Convergence'
        # Return
        trainingData -> 2d Array of float
        '''
        trainingData = []
        for s in range(self.data.shape[0]):
            print("Processing sample %d" %s)
            for i in range(self.data.shape[1]):
                movingIndex = dataLen
                while not tools.isConverged(self.data[s, i, (movingIndex-dataLen):movingIndex], trigger):
                    trainingData.append( self.data[s, i, (movingIndex-dataLen):movingIndex] )
                    movingIndex += 1
        return np.array(trainingData)

def showG14Std():
    miniG14 = MiniDataset(g14)
    for i in range(miniG14.data.shape[0]):
        std = np.std(miniG14.data[i, :, 0:200], axis=0)
        std = np.mean(std)
        print(str(i) + " std= " + str(std))


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

    # trainingData = miniG14.getTrainingData(15, 0.001)
    # print(trainingData.shape)

if __name__ == "__main__":
    main()
