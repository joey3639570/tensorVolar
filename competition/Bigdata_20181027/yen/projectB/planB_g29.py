import planB
import TypeInfo
import readDatasets2 as rd

def main():
    typeInfo = TypeInfo.G30()
    rd.Datasets.set_root('/projectB/1027B')
    dataset = rd.Datasets(typeInfo.DATASET_NAME)
    miniDataset = rd.MiniDataset(typeInfo, dataset)
    
    rd.Datasets.set_root('/projectB/1027B_test')
    t_dataset = rd.Datasets(typeInfo.DATASET_NAME, test=True)
    t_miniDataset = rd.MiniDataset(typeInfo, t_dataset)
    

    avr, std = planB.autoMatch(typeInfo, 20, miniDataset=miniDataset, testDataset=t_miniDataset, show=True, startIndex=10)
    print("avr: {}, std: {}".format(avr, std))

if __name__ == "__main__":
    main()
