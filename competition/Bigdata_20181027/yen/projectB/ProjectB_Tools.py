import numpy as np

def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'same')
    return sma

def isConverged_std(typeInfo, section):
    std = np.std(section)
    if std<typeInfo.EPSILON and dataList[0] > typeinfo.CONVERGE_MIN_TEMPERATURE:
        return True
    return False

def isConverged_abs_mean(typeInfo, section, filter=True):
    section_avr = np.mean(section)
    epsilon = np.mean(abs(section - section_avr))
    if epsilon < typeInfo.CONVERGE.EPSILON and ((not filter) or section_avr > typeInfo.CONVERGE_MIN_TEMPERATURE):
        return True
    return False

def isConverged_relative(typeInfo, section, filter=True):
    section_avr = np.mean(section)
    middle = typeInfo.CONVERGE.SAMPLING_LEN // 2
    f_center = middle//2
    r_center = f_center + middle
    ff_avr = np.mean(section[0:f_center])
    fr_avr = np.mean(section[f_center:middle])
    rf_avr = np.mean(section[middle:r_center])
    rr_avr = np.mean(section[r_center:-1])
    epsilon = abs((rr_avr-rf_avr) - (fr_avr-ff_avr))
    if epsilon < typeInfo.CONVERGE.EPSILON_RELATIVE and ((not filter) or section_avr > typeInfo.CONVERGE_MIN_TEMPERATURE):
        return True
    return False



def getConvIndex(typeInfo, dataList, multi=False):
    '''
    Return the index of converge point
    Parameters:
        typeInfo -> The type info object
        dataList -> 1d Array sensor value
        moving_len -> perform moving average to the input data
        cool_len -> In multi mode, it will prevent to output too close index
    '''
    moving_len = typeInfo.MOVING_LEN
    cool_len = typeInfo.COOL_LEN
    sampleLen = typeInfo.CONVERGE.SAMPLING_LEN
    dataList = moving_average(dataList, window=moving_len)
    if multi:
        multiIndex = []
        preState = False
        preIndex = 0
    for i in range(len(dataList) - sampleLen):
        if multi:
            if preState != isConverged_abs_mean(typeInfo, dataList[i:i+sampleLen], filter=False):
                if i - preIndex > cool_len:
                    preState = not preState
                    multiIndex.append(i)
                    preIndex = i
        else:
            if isConverged_abs_mean(typeInfo, dataList[i:i+sampleLen]):
                return i + typeInfo.CONVERGE.REFERENCE_INDEX
    if multi:
        return np.array(multiIndex)
    print("Error! DataList can not converge")

class BatchGenerator:
    def __init__(self, trainingData, startTime, endTime):
        self.trainingData = trainingData
        self.startTime = startTime
        self.endTime = endTime
        self.sampleLen = self.trainingData.shape[0]
        self.nowIndex = 0

    def shuffle(self):
        '''
        prameter:
        dataList -> 2d Array has shape[sample, len] 
        '''
        idx = np.arange(0 , self.trainingData.shape[0])
        np.random.shuffle(idx)
        data_shuffle = [self.trainingData[i] for i in idx]
        self.trainingData = np.array(data_shuffle)

    def getBatch(self, batchSize):
        '''
        Return a batch and labels
        [trainingData, startTime, endTime]
        trainingData shape = [batchSize, len]
        startTime = [batchSize]
        endTime = [batchSize]
        '''
        if self.nowIndex + batchSize > self.sampleLen:
            # Last index exceed
            s = self.sampleLen-batchSize - 1 # Start Time
            e = self.sampleLen - 1           # End Time
            self.nowIndex = 0
        else:
            s = self.nowIndex                # Start Time
            e = self.nowIndex + batchSize    # End Time
            out = np.squeeze(self.trainingData[s:e])
        self.nowIndex += batchSize
        return out, self.startTime[s:e], self.endTime[s:e]

