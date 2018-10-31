import numpy as np
import planB
import readDatasets2 as rd
import ProjectB_Tools as tools
import TypeInfo
from yen import show_data_chart as plt
import scipy.stats as stats

def main():
    typeInfo = TypeInfo.G35()
    dataset = rd.Datasets(typeInfo.DATASET_NAME)
    miniDataset = rd.MiniDataset(typeInfo, dataset)
    # Calculate the model
    #plot = plt.Plot(400, 70, 330, 150)
    #plot.appendData(miniDataset.data[0, 0])
    #plot.plot()
    numOfSample = miniDataset.data.shape[0]
    bufferSize = 350
    firstClimb = np.zeros([numOfSample, bufferSize], dtype=np.float32)
    secondClimb = np.zeros([numOfSample, bufferSize], dtype=np.float32)
    #For first conv temp vs second conv temp
    '''
    first_temp = []
    second_temp = []
    third_temp = []
    fourth_temp = []
    conv_index1 = []
    conv_index2 = []
    '''
    for sampleIdx in range(numOfSample):
        conv_index = tools.getConvIndex(typeInfo, miniDataset.data[sampleIdx, typeInfo.SENSOR_INDEX], multi=True)
        print("Sample {} conv at {}".format(sampleIdx, conv_index))
        '''
        for i in range(4):
            print("Sample {} Temperature {}  :: {}".format(sampleIdx , i, miniDataset.data[sampleIdx, 0, conv_index[i]]))
            if i % 4 == 0:
                first_temp.append(miniDataset.data[sampleIdx, 0, conv_index[i]])
                conv_index1.append(conv_index[i])
            if i % 4 == 1:
                second_temp.append(miniDataset.data[sampleIdx, 0, conv_index[i]])
                conv_index2.append(conv_index[i])
            if i % 4 == 2:
                third_temp.append(miniDataset.data[sampleIdx, 0, conv_index[i]])
            if i % 4 == 3:
                fourth_temp.append(miniDataset.data[sampleIdx, 0, conv_index[i]])    
            '''
        firstClimb[sampleIdx] = np.resize(miniDataset.data[sampleIdx, typeInfo.SENSOR_INDEX, 0:conv_index[1]], bufferSize)

        secondClimb[sampleIdx] = np.resize(miniDataset.data[sampleIdx, typeInfo.SENSOR_INDEX, conv_index[1]:conv_index[3]], bufferSize)
    print(firstClimb)

    firstAligned = planB.alignInHorizontal(typeInfo, miniDataset, data=firstClimb, useSpecifiedData=True)
    secondAligned = planB.alignInHorizontal(typeInfo, miniDataset, data=secondClimb, useSpecifiedData=True)

    firstModel = planB.calModel(firstAligned)
    firstModelConvIndex = tools.getConvIndex(typeInfo, firstModel)

    secondModel = planB.calModel(secondAligned)
    secondModelConvIndex = tools.getConvIndex(typeInfo, secondModel)
    
    chartA = plt.Plot(400, 70, 350, 150)
    for i in range(firstAligned.shape[0]):
        chartA.appendData(firstAligned[i])
    #chartA.appendData(firstModel)
    chartA.plot()
    
    chartB = plt.Plot(400, 70, 350, 150)
    for i in range(secondAligned.shape[0]):
        chartB.appendData(secondAligned[i])
    #chartB.appendData(secondModel)
    chartB.plot()
    '''
    print("Correlation between first temp & third temp :: ",stats.pearsonr(first_temp, third_temp))
    print("Correlation between first temp & fourth temp :: ",stats.pearsonr(first_temp, fourth_temp))
    print("Correlation between second temp & third temp :: ",stats.pearsonr(second_temp, third_temp))
    print("Correlation between second temp & fourth temp :: ",stats.pearsonr(second_temp, fourth_temp))
    
    print("Range of First temp :: ",np.max(first_temp)-np.min(first_temp))
    print("Range of Second temp :: ",np.max(second_temp)-np.min(second_temp))
    print("Range of Third temp :: ",np.max(third_temp)-np.min(third_temp))
    print("Range of Fourth temp :: ",np.max(fourth_temp)-np.min(fourth_temp))

    print("Range of First conv index :: ",np.max(np.array(conv_index1)-np.array(conv_index2))-np.min(np.array(conv_index1)-np.array(conv_index2)))
    print("Good data :: ", miniDataset.goodData)
    '''
    #print(firstModelConvIndex)
    #print(secondModelConvIndex)

if  __name__ == "__main__":
    main()
