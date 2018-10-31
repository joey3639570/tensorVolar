from joey import get_data
import numpy as np
from termcolor import colored
import sys
class Plot():
    def __init__(self, upperBound, lowerBound, chartWidth, chartHeight):
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.chartWidth = chartWidth
        self.chartHeight = chartHeight
        self.maps = np.zeros(shape=(chartWidth, chartHeight), dtype=np.uint8)
        self.numOfData = 1
        self.maxNumOfData = 8
        self.colorSet = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
        self.labels = []

    def appendData(self, dataList, label=0):
        # assert(not self.numOfData > self.maxNumOfData), "Too much data in one chart!"
        # self.dataList = average(dataList, dataList.shape[0]//self.chartWidth)
        self.dataList = samplize(dataList, self.chartWidth)
        self.labels.append(label)
        for w in range(self.chartWidth):
            y = int(self.chartHeight*(self.dataList[w]-self.lowerBound)//(self.upperBound-self.lowerBound))
            if y >= self.chartHeight:
                y = self.chartHeight-1
            self.maps[w, y] = self.numOfData
        self.numOfData += 1

    def appendLine(self, value, ver=True):
        if ver:
            for h in range(self.chartHeight):
                self.maps[value, h] = 99
        else:
            for w in range(self.chartWidth):
                self.maps[w, value] = 99


    def plot(self):
        for h in range(self.chartHeight):
            hh = self.chartHeight - h - 1
            print("%##2.2f" %((self.upperBound-self.lowerBound)*hh/self.chartHeight+self.lowerBound), end='')
            for w in range(self.chartWidth):
                if self.maps[w, hh] == 0:
                    print(' ', end='')
                else:
                    print(colored('*', self.colorSet[self.maps[w, hh]]), end='')
            print(' ')
    def plotColorful(self):
        for h in range(self.chartHeight):
            hh = self.chartHeight - h - 1
            print("%##2.2f" %((self.upperBound-self.lowerBound)*hh/self.chartHeight), end='')
            for w in range(self.chartWidth):
                if self.maps[w, hh] == 0:
                    print(' ', end='')
                else:
                    sys.stdout.write(u"\u001b[38;5;" + str(16 + 2*self.maps[w, hh]) + "m" + '*')
                    # print(colored('*', self.colorSet[self.maps[w, hh]]), end='')
            print(' ')


    def clear(self):
        for h in range(self.chartHeight):
            for w in range(self.chartWidth):
                maps[w, h] = 0

    def label(self):
        for l in range(self.numOfData):
            out = ('Label '+ str(self.labels[l-1]) + ' = '+ str(self.colorSet[l]))
            print(colored(out, self.colorSet[l]))

def samplize(data_list, outputSize):
    inputSize = data_list.shape[0]
    data_output = []
    for i in range(outputSize):
        data_output.append(data_list[i*inputSize//outputSize])
    return np.array(data_output)

def show_data(data_list, chart_height=200):
    '''
    show data in stars.
    paras=
    data_list:    1 dimention input data.
    chart_height: max length of height(y direction) 
    '''
    max_num = np.max(data_list)
    for row in data_list:
        y = int(chart_height*row/max_num)
        for i in range(y):
            print(" ", end='')
        print("*", end='')
        for i in range(chart_height-y-1):
            print(" ", end='')
        print(row)

def show_data_trans(data_list, chart_width, chart_height):
    data_list = average(data_list, data_list.shape[0]//chart_width)
    print(data_list)
    data_length = data_list.shape[0]
    max_num = np.max(data_list)
    min_num = np.min(data_list)
    print('min= ', min_num, ',max= ', max_num)
    maps = np.zeros(shape=(int(data_length), int(chart_height)), dtype=np.uint8)
    for i in range(data_length):
        y = int(chart_height*(data_list[i]-min_num)/(max_num-min_num))
        if y >= chart_height:
            y = chart_height - 1
        # print('raw= ',data_list[i], 'trans= ', y) 
        maps[i, y] = 1
    for h in range(chart_height):
        print("%2.2f" %(max_num-(max_num-min_num)*h/chart_height), end='')
        for w in range(chart_width):
            if maps[w, h] == 0:
                print(' ', end='')
            else:
                print('*', end='')
        print(' ')

def average(arr, num):
    """
    Average pooling with each 'num' parts.
    :param arr: Input array
    :param num: Slice into 'num' part.
    :return: Averaged array
    """
    size = int(arr.shape[0]//num)
    ave = np.zeros(size)
    for i in range(0, size):
        ave[i] = np.mean(arr[i*num:(i+1)*num])
    return ave

def main():
    train_dir = '/home/t125501/workplace/projectA/920A'

    data_filenames, label = get_data.get_list(train_dir)
    data, label = get_data.get_batch(data_filenames, label ,16)
    chart = Plot(0.5, -0.5,  1300, 500)
    for i in range(8):    
        chart.appendData(data[i], label[i])
    chart.plot()
    chart.label()
    #data = get_data.get_averaged_data(data, 2560)
    #show_data_trans(data[0], 130, 50)
    #print(data)
    #print(label)
    #print("data shape=", data.shape)
    #print("label shape=", label.shape)
    #print(colored('hello', 'red'), colored('world', 'green'))

if __name__ == "__main__":
    main()
