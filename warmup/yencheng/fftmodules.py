import numpy as np
import tools
from scipy import stats

def calFFT(dataList):
    return np.fft.fft(dataList)

def calFFT_with_frq(dataList, frq):
    y = tools.interpolation(dataList, frq)
    print('y shape=', y.shape)
    output = tools.slice(np.fft.fft(y), 0, 0.5)
    return y, output

def calFFT_with_frq_full(dataList, frq):
    y = tools.full_interpolation(dataList, frq)
    return y, np.fft.fft(y)

def fft_set1(dataList):
    """
    Using 5 average pooling first, then do find_real_wave to simplify the number of variables.
    Calculate fft transformation with frequency
    :param dataList:
    :return:
    """
    SLICE = 1
    MOVING = 5
    TRIGGER = 1e-8
    FRQ = 200
    CUTTER = 1

    # output = tools.centered_moving_average(sample.dataList[:, varID], MOVING)
    # output = stats.zscore(output)
    output = tools.average(dataList, MOVING)
    output = tools.find_real_wave(output, triggerValue=TRIGGER)
    # print(output.shape)
    output = tools.slice(output, 0, SLICE)
    y, ffty = calFFT_with_frq_full(output, FRQ)
    ffty = (abs(ffty))
    ffty = ffty[0: len(ffty) // 2]
    # ffty = tools.softmax(ffty)
    ffty = tools.centered_moving_average(ffty, 5)
    ffty = stats.zscore(ffty)
    # ffty = tools.average(ffty, 2)
    fftx = np.arange(len(y))
    freq = np.arange(len(ffty))
    return y, ffty

def fft_set2(dataList):
    SLICE = 1
    MOVING = 5
    TRIGGER = 1e-8
    FRQ = 400
    CUTTER = 1

    # output = tools.centered_moving_average(sample.dataList[:, varID], MOVING)
    # output = stats.zscore(output)
    output = tools.average(dataList, 100)
    y = tools.full_interpolation_simple(output, FRQ)
    ffty = np.fft.fft(y)
    ffty = (abs(ffty))
    ffty = ffty[0: len(ffty) // 2]
    # ffty = tools.softmax(ffty)
    # ffty = tools.centered_moving_average(ffty, 5)
    # ffty = stats.zscore(ffty)
    # ffty = tools.average(ffty, 2)
    # fftx = np.arange(len(y))
    # freq = np.arange(len(ffty))
    return output, ffty
    '''
    plt.subplot(8, 40 / cutter, int(80 * varID / cutter + sampleCount / cutter))
    plt.plot(fftx, y)
    # plt.axis([0, FRQ, 0, 0.00004])

    plt.subplot(8, 40 / cutter, int(80 * varID / cutter + sampleCount / cutter + 40 / cutter))
    plt.plot(freq, ffty)
    # plt.axis([0, FRQ, 0, 0.0005])
    '''