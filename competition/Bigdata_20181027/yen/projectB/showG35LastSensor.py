from yen import show_data_chart as plt
import readDatasets2 as rd
import TypeInfo
import planB

def main():
    typeInfo = TypeInfo.G35()
    g35 = rd.Datasets(typeInfo.DATASET_NAME)
    #g35 = rd.Datasets('G35')
    miniG35 = rd.MiniDataset(typeInfo, g35)
    typeInfo.SENSOR_INDEX = 2
    alignedSample = planB.alignInHorizontal(typeInfo, miniG35)

    chart = plt.Plot(400, 70, 330, 150)
    for i in range(alignedSample.shape[0]):
        chart.appendData(alignedSample[i])
    chart.plotColorful()
    '''
    chart = plt.Plot(400, 70, 600, 150)
    print(g35.samples[0].dataList[0].shape)
    chart.appendData(g35.samples[0].dataList[0], 0)
    chart.plotColorful()
    chart = plt.Plot(400, 70, 600, 150)
    for i in range(miniG35.data.shape[0]):
        chart.appendData(miniG35.data[i, 0], 0)
    chart.plotColorful()
    '''
if __name__ == "__main__":
    main()
