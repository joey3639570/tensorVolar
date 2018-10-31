import numpy as np
from scipy import stats
from pyhht.emd import EMD
import tools
import fftmodules
def emd_transformation(dataList):
    dataList = tools.average(dataList, 100)
    decomposer = EMD(dataList)
    imfs = decomposer.decompose()

    _, emd0 = fftmodules.fft_set1(imfs[0])
    #emd0 = np.abs(imfs[0])
    #emd0 = (imfs[0])
    print('emd0_p= ', emd0.shape)
    #emd0 = tools.average(arr=emd0, num=3)
    #emd0 = tools.moving_average(a=emd0, n=5)
    print('emd0_a= ', emd0.shape)
    _, emd1 = fftmodules.fft_set1(imfs[1])
    #emd1 = np.abs(imfs[1])
    #emd1 = (imfs[1])
    #emd1 = tools.average(arr=emd1, num=3)
    #emd1 = tools.moving_average(a=emd1, n=5)
    #emd2 = np.abs(imfs[2])
    _, emd2 = fftmodules.fft_set1(imfs[2])
    #emd2 = (imfs[2])
    #emd2 = tools.average(arr=emd2, num=3)
    #emd2 = tools.moving_average(a=emd2, n=5)
    return [emd0, emd1, emd2]

    # return [imfs[0], imfs[1], imfs[2]]

def split_train_test(imfs1, imfs2, imfs3):
    imfs1 = stats.zscore(imfs1)
    imfs2 = stats.zscore(imfs2)
    imfs3 = stats.zscore(imfs3)
    imfs1_train = []
    imfs1_test = []
    imfs2_train = []
    imfs2_test = []
    imfs3_train = []
    imfs3_test = []
    y_train = []
    y_test = []
    test_index = [1, 5, 9, 13, 17, 21, 25, 30, 35, 38]
    for i in range(40):
        c = 0
        for j in range(len(test_index)):
            if(i == test_index[j]):
                c = 1
                imfs1_test.append(imfs1[i])
                imfs2_test.append(imfs2[i])
                imfs3_test.append(imfs3[i])
                y_test.append(y[i])
        if(c == 0):
            imfs1_train.append(imfs1[i])
            imfs2_train.append(imfs2[i])
            imfs3_train.append(imfs3[i])
            y_train.append(y[i])
    imfs1_train = np.array(imfs1_train)
    imfs1_test = np.array(imfs1_test)
    imfs2_train = np.array(imfs2_train)
    imfs2_test = np.array(imfs2_test)
    imfs3_train = np.array(imfs3_train)
    imfs3_test = np.array(imfs3_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return imfs1_train, imfs1_test, imfs2_train, imfs2_test, imfs3_train, imfs3_test, y_train, y_test
