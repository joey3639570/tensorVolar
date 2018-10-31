from yen import show_data_chart as plt
import readDatasets2 as rd
import TypeInfo
import planB
import numpy as np
def main():
    typeInfo = TypeInfo.G29()
    rd.Datasets.set_root('/projectB/1027B')
    dataset = rd.Datasets(typeInfo.DATASET_NAME)
    np.set_printoptions(threshold=np.inf)
    print(dataset.samples[0].dataList[0])
    #exit()
    miniDataset = rd.MiniDataset(typeInfo, dataset)
    # Calculate the model
    alignedSample = planB.alignInHorizontal(typeInfo, miniDataset)
    
    chart = plt.Plot(400, 70, 100, 50)
    for i in range(alignedSample.shape[0]):
        chart.appendData(alignedSample[i])
    chart.plotColorful()

if __name__ == "__main__":
    main()
