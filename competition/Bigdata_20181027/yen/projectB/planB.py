import numpy as np
import readDatasets2 as rd
import TypeInfo
import ProjectB_Tools as tools
import datetime
from yen import show_data_chart as plt

def alignInHorizontal(typeInfo, miniDataset, data=None, useSpecifiedData=False):
    '''
    This function will output a 2d array data which is aligned according to the 
    sensor temperature specified in typeinfo.
    If useSpecifiedData set True, it will extract resource from 'data' array
    '''
    # Get offset distance to align
    if not useSpecifiedData:
        # data = miniDataset.data[typeInfo.GOOD_SAMPLES, typeInfo.SENSOR_INDEX]
        data = miniDataset.data[:, typeInfo.SENSOR_INDEX]

    print("Model data shape= {}".format(data.shape))
    baseTemperature = data[typeInfo.BASE_SAMPLE_INDEX, typeInfo.ALIGN_INDEX]
    offsetIndex = np.argmin( a=abs(data[:,0:typeInfo.SEARCH_RANGE]-baseTemperature), axis=1 )
    # Align the array
    minIndex = np.min(offsetIndex)
    length = data.shape[1] - np.max(offsetIndex) + np.min(offsetIndex)
    output = np.ones([len(offsetIndex), length], np.float32)
    start = np.array(offsetIndex-minIndex, dtype=int)
    end = np.array(start+length, dtype=int)
    # print(str(start) + ", \n" + str(end))
    for si in range(len(offsetIndex)):
        output[si] = data[si, start[si]:end[si]]
    return output

def calModel(data):
    '''
    This function will calculate the model from a aligned input data.
    Parameter:
    data -> 2d Array with shape [sample, value] each sample only have 1 sensor.
    '''
    output = np.mean(data, axis=0)
    return output

def getPlanB_TrainingData(miniDataset, dataLen=20, sensorIndex=[0], startIndex=0):
    '''
    This function output training data samples from minidataset
    Parameters:
        miniDataset -> 
        sensor_index -> set of sensors to generate training data
        (Depend on size of miniDataset)
    '''
    trainingData = []
    startTime = []
    endTime = []
    for s in range(miniDataset.data.shape[0]):
        #print("Processing sample%d" %s)
        for idx in sensorIndex:
            trainingData.append(miniDataset.data[s,idx,startIndex:startIndex+dataLen])
            startTime.append(miniDataset.startTime[s])
            endTime.append(miniDataset.endTime[s])
    trainingData = np.array(trainingData)
    startTime = np.array(startTime)
    endTime = np.array(endTime)
    return trainingData, startTime, endTime

def calDataDiff_abs_mean(data1, data2):
    # print("Len= "+str(data1.shape) + ", " + str(data2.shape))
    assert( len(data1) == len(data2) ), "Length of the two array must be equal."
    epsilon = np.mean(abs(data1 - data2))
    return epsilon

def sampleFitModel(inputs, model):
    diff = np.zeros(len(model), dtype=np.float32)
    input_len = len(inputs)
    #print("Input shape= " + str(inputs.shape))
    #print("Model shape= " + str(model.shape))
    for i in range(1, len(model)-input_len):
        if i < input_len:
            diff[i] = calDataDiff_abs_mean(inputs[input_len-i-1:input_len-1], model[0:i])
        else:
            diff[i] = calDataDiff_abs_mean(inputs, model[i:i+input_len])
    minIndex = np.argmin(diff)
    return minIndex + input_len

def plotModel(typeInfo):
    dataset = rd.Datasets(typeInfo.DATASET_NAME)
    miniDataset = rd.MiniDataset(dataset)
    alignedSample = alignInHorizontal(typeInfo, miniDataset)
    model = calModel(alignedSample)
    chart = plt.Plot(400, 70, 330, 150)
    # for i in range(output.shape[0]):
    chart.appendData(model, 0)
    chart.plot()

def autoMatch(typeInfo, dataLen, miniDataset=None, testDataset=None, show=False, startIndex=0, k=0):
    if(miniDataset == None):
        dataset = rd.Datasets(typeInfo.DATASET_NAME)
        miniDataset = rd.MiniDataset(typeInfo, dataset)
    # Calculate the model
    alignedSample = alignInHorizontal(typeInfo, miniDataset)
    '''
    plot = plt.Plot(400, 70, 330, 150)
    for i in range(alignedSample.shape[0]):
        plot.appendData(alignedSample[i])
    plot.plotColorful()
    '''
    model = calModel(alignedSample)
    model_conv_index = tools.getConvIndex(typeInfo, model)
    # Generate training data and split into batches
    trainingData, startTime, endTime = getPlanB_TrainingData(testDataset, dataLen=dataLen, sensorIndex=[0], startIndex=startIndex)
    batches = tools.BatchGenerator(trainingData, startTime, endTime)
    # print("TrainingData shape= " + str(trainingData.shape))
    batches.shuffle()
     
    M_model = dataLen/(model[dataLen]-model[0])
    
    diff = []
    for i in range(trainingData.shape[0]):
        batch, st, et = batches.getBatch(1)
        offsetIndex = sampleFitModel(batch, model)


        T_model = dataLen/(batch[-1]-batch[0])
        slope_offset = (M_model-T_model)*k

        predict = st + datetime.timedelta(seconds=int(model_conv_index - offsetIndex+typeInfo.MODEL_AVR_OFFSET)*60)
        # print("Time diff= " + str(model_conv_index - offsetIndex))
        # print("Sample {}, predict= {}, ans= {}".format(i, str(predict[0]), str(et[0])))
        diff.append( (et[0] - predict[0]).total_seconds()//60 )
    if show:
        print("Time diff= " + str(diff))

    avr = np.mean(diff)
    std = np.std(diff)
    # print("avr: {}, std: {}".format(avr, std))
    return avr, std

def getBadSampleIndex(diff, avr, std, scope=1.5):
    out = np.ones(len(diff), bool)
    accuracy = len(diff)
    for i, d in enumerate(diff):
        if abs(d-avr) > scope*std:
            out[i] = False
            accuracy -= 1
    accuracy /= len(diff)
    return out, accuracy

def main():
    typeInfo = TypeInfo.G57()
    dataset = rd.Datasets(typeInfo.DATASET_NAME)
    miniDataset = rd.MiniDataset(typeInfo, dataset)
    # Calculate the model
    alignedSample = alignInHorizontal(typeInfo, miniDataset)
    '''
    plot = plt.Plot(400, 70, 330, 150)
    for i in range(alignedSample.shape[0]):
        plot.appendData(alignedSample[i])
    plot.plotColorful()
    '''
    #model = calModel(alignedSample)
    #model_conv_index = tools.getConvIndex(typeInfo, model)
    typeInfo.SENSOR_INDEX = 0    
    alignedSample = alignInHorizontal(typeInfo, miniDataset)
    model1 = calModel(alignedSample)
    typeInfo.SENSOR_INDEX = 1
    alignedSample = alignInHorizontal(typeInfo, miniDataset)
    model2 = calModel(alignedSample)
    typeInfo.SENSOR_INDEX = 2
    alignedSample = alignInHorizontal(typeInfo, miniDataset)
    model3 = calModel(alignedSample)
    #alignedSample = alignInHorizontal(typeInfo, None, data=[model1, model2, model3], useSpecifiedData=True)

    plot = plt.Plot(400, 70, 330, 150)
    #for i in range(alignedSample.shape[0]):
    plot.appendData(model1)
    plot.appendData(model2)
    plot.appendData(model3)

    plot.plotColorful()

    maxSize = np.max([model1.shape[0], model2.shape[0], model3.shape[0]])
    model1 = np.resize(model1, maxSize)
    model2 = np.resize(model2, maxSize)
    model3 = np.resize(model3, maxSize)

    model = calModel([model1, model2, model3])
    model_conv_index = tools.getConvIndex(typeInfo, model)

    # Generate training data and split into batches
    trainingData, startTime, endTime = getPlanB_TrainingData(miniDataset, dataLen=20, sensorIndex=[0])
    
    batches = tools.BatchGenerator(trainingData, startTime, endTime)
    batches.shuffle()
    
    diff = []
    for i in range(trainingData.shape[0]):
        batch, st, et = batches.getBatch(1)
        offsetIndex = sampleFitModel(batch, model)
        predict = st + datetime.timedelta(seconds=int(model_conv_index - offsetIndex+typeInfo.MODEL_AVR_OFFSET)*60)
        # print("Time diff= " + str(model_conv_index - offsetIndex))
        # print("Sample {}, predict= {}, ans= {}".format(i, str(predict[0]), str(et[0])))
        diff.append( (et[0] - predict[0]).total_seconds()//60 )
    print("Time diff= " + str(diff))

    avr = np.mean(diff)
    std = np.std(diff)
    print("avr: {}, std: {}".format(avr, std))
    badSampleIndex, accuracy = getBadSampleIndex(diff, avr, std, 1)
    print("Bad Sample Index")
    for item in badSampleIndex:
        print(item, end=', ')
    print("\nAccuracy= {}".format(accuracy))
    accuracy = 0
    for item in diff:
        if(abs(item) <= 5):
            accuracy += 1
        #print(abs(item))
    accuracy /= len(diff)
    print("Accuracy= {}".format(accuracy))

    # print("Bad Samples: {}".format(badSampleIndex)) 
    return avr, std

if __name__ == '__main__':
    main()
