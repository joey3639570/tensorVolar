from yen import show_data_chart as plt
import readDatasets2 as rd
import TypeInfo
import planB
def main():
    typeInfo = TypeInfo.G44()
    rd.Datasets.set_root('/projectB/1027B')
    dataset = rd.Datasets(typeInfo.DATASET_NAME)
    #miniDataset = rd.MiniDataset(typeInfo, dataset)
    # Calculate the model
    #alignedSample = planB.alignInHorizontal(typeInfo, miniDataset)
    
    chart = plt.Plot(400, 70, 100, 50)
    for i in range(dataset.samples[0].dataList.shape[0]):
        chart.appendData(dataset.samples[0].dataList[i])
    chart.appendLine(15)
    chart.appendLine(30)
    chart.appendLine(45)

    chart.plotColorful()

if __name__ == "__main__":
    main()
