import numpy as np
from scipy import stats

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def centered_moving_average(array, size):
    sumed = np.zeros(len(array)-size)
    for i in range(size):
        sumed += array[i:-size+i]
    return sumed/size


def average(arr, num):
    """Average pooling with each 'num' parts"""
    size = int(arr.shape[0]//num)
    ave = np.zeros(size)
    for i in range(0, size):
        ave[i] = np.mean(arr[i*num:(i+1)*num])
    return ave


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def find_real_wave(dataList, triggerValue):
    """Create a new list which extract the interesting part. It will return a new array contain x and y."""
    preValue = 0
    output = []
    lineCounter = 0
    movingAvr = 0
    for line in dataList:
        movingAvr = 0.9*movingAvr + 0.1*line
        if line - preValue > triggerValue:
            output.append([lineCounter, line])
        preValue = line
        lineCounter += 1
    output = np.array(output)
    return output


def slice(dataList, startRate=0, endRate=1):
    """Slice a dataset according to the specified portion"""
    maxLength = len(dataList)
    return dataList[int(maxLength*startRate): int((maxLength-1)*endRate)]

def interpolation(dataList, xSize):
    if dataList.shape[1] == 2:
        return np.interp(np.arange(xSize), xp=dataList[:, 0], fp=dataList[:, 1])

def full_interpolation(dataList, xSize):
    x = np.linspace(0, 1, xSize)
    y = x
    oriCount = 0
    maxX = dataList[len(dataList[:, 0]) - 1, 0]
    preP = dataList[oriCount]
    preP[0] *= xSize / maxX
    nextP = dataList[oriCount + 1]
    nextP[0] *= xSize / maxX
    for i in range(0, xSize - 1):
        y[i] = preP[1] + (i - preP[0]) * (nextP[1] - preP[1]) / (nextP[0] - preP[0])
        if i >= nextP[0] and oriCount + 2 < len(dataList[:, 0]):
            oriCount += 1
            preP = dataList[oriCount]
            preP[0] *= xSize / maxX
            nextP = dataList[oriCount + 1]
            nextP[0] *= xSize / maxX
    y[xSize - 1] = dataList[len(dataList[:, 0]) - 1, 1]
    return y

def full_interpolation_simple(dataList, xSize):
    # x = np.linspace(0, 1, xSize)
    y = np.interp(np.arange(xSize), xp=np.arange(len(dataList)), fp=dataList)
    return y

def zscore(data):
    return stats.zscore(data)