import planB
import TypeInfo
import readDatasets2 as rd
import numpy as np

def main():    
    typeInfo = TypeInfo.G14()
    dataset = rd.Datasets(typeInfo.DATASET_NAME)

    epsilons = np.arange(0.069, 0.001, -0.001)
    sampleLens = np.arange(5, 25, 1)
    alignedIndeices = np.arange(10, 50, 5)
    sensorIndeices = np.arange(0, 5, 1)
    dataLens = np.arange(5, 35, 5)
    best = 9999

    for epsilon in epsilons: # EPSILON
        for sampleLen in sampleLens: # SAMPLE_LEN
            #for aligned in alignedIndeices: # ALIGNED
                miniDataset = rd.MiniDataset(typeInfo, dataset)
            #    for sensor in sensorIndeices:
            for dataLen in dataLens: 
                typeInfo.CONVERGE = TypeInfo.Converge(sample_len=sampleLen, epsilon=epsilon, ref_index='middle')
                typeInfo.ALIGN_INDEX = aligned
                typeInfo.SENSOR_INDEX = sensor
                avr, std = planB.autoMatch(typeInfo, dataLen, miniDataset)
                print("Log: " + str([epsilon, sampleLen, aligned, sensor, dataLen]) + "avr: " + str(avr) + ", std: " + str(std))

                if std < best:
                    configuration = [epsilon, sampleLen, aligned, sensor, dataLen]

    print("Best Epsilon :: ", configuration[0])
    print("Best Sample len :: ", configuration[1])
    print("Best Aligned index :: ", configuration[2])
    print("Best Sensor index :: ", configuration[3])
    print("Best data len :: ", configuration[4])

if __name__ == '__main__':
    main()
