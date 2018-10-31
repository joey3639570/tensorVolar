import read_datatest as datasets
from yen import show_data_chart as plt
import os

JobDir = '/home/t125501/workplace/yen/projectB'
G14Dir = os.path.join(JobDir, 'G14')
G35Dir = os.path.join(JobDir, 'G35')
G57Dir = os.path.join(JobDir, 'G57')



def get_batch(data_list, label_list, batch_size):
    data_batch = []
    idx = np.arange(0, len(data_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    shuffled_data = [data_list[i] for i in idx]
    shuffled_lable = [label_list[i] for i in idx]
    return shuffled_data

def get_batch(data_list, batch_size):
    data_batch = []
    idx = np.arange(0, len(data_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    shuffled_data = [data_list[i] for i in idx]
    return shuffled_data


def main():
    g14 = datasets.getG14Dataset(G14Dir)
    g35 = datasets.getG14Dataset(G35Dir)
    g57 = datasets.getG14Dataset(G57Dir)
    # print(g14[0].dataList[0])
    
    #g14 = datasets.getG14Dataset(G14Dir)
    #print(g14[0].dataList[0])
    chart = plt.Plot(400, 70, 330 ,150)
    #chart = plt.Plot(400, 70, 500 ,150)
    #for i in range(30):
    #    chart.appendData(g14[i].dataList[0], g14[i].dataLabel[0])
    for i in range(25):
        #for j in range(5):
        chart.appendData(g14[i].dataList[i], g14[i].dataLabel[i])
 
    #chart.appendData(g14[0].dataList[0], 0)

    #chart.plot()
    chart.plotColorful()
    #chart.label()


if __name__ == "__main__":
    main()
