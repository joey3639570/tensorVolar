import os
import read_datatest as rd
import numpy as np
from yen import show_data_chart as plt
#from scipy import stats
JobDir = '/home/t125501/workplace/csvprojectB'
datasetsName = ['G14', 'G35', 'G57']

def main():
    G14Dir = os.path.join(JobDir, 'G14')
    G35Dir = os.path.join(JobDir, 'G35')
    G57Dir = os.path.join(JobDir, 'G57')
    
    g14_datasets = rd.getG14Dataset(G14Dir)
    g35_datasets = rd.getG35Dataset(G35Dir)
    g57_datasets = rd.getG57Dataset(G57Dir)
    
    datasets = [g14_datasets, g35_datasets, g57_datasets]
    
    for dataset in datasets:
        deltaT = []
        for sample in dataset:
            maxT = []
            for j in range(sample.dataList.shape[0]):
                mean = np.mean(sample.dataList[j])
                maxV = np.max(sample.dataList[j])
                #minV = np.min(sample.dataList[j])
                #if np.abs(mean - maxV) < 200:
                if maxV < 400 and maxV > 200:
                    maxT.append(maxV)
            #print(maxT)
            sensor_avr_max = np.max(maxT)
            sensor_avr_min = np.min(maxT)
            deltaT.append(sensor_avr_max - sensor_avr_min)
        #avr = np.mean(deltaT)
        print(deltaT)
        maxBound = np.max(deltaT)
        minBound = np.min(deltaT)
        print("\nDataset:")
        chart = plt.Plot(maxBound+20, minBound-20, 300, 100)
        chart.appendData(np.array(deltaT), 0)
        chart.plot()
        #print(deltaT)
        #print("max")
        #print(maxT)

if __name__ == "__main__":
    main()
