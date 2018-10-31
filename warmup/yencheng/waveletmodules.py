import numpy as np
import tools
import pywt

def wavelet(dataList, waveletMode='db2'):
    moving_average_size = 500
    # row = np.array([], np.float32)
    # averaged = tools.centered_moving_average(dataList, moving_average_size)
    # afterwt = pywt.wavedec(averaged, waveletMode)
    afterwt = pywt.wavedec(dataList, waveletMode)
    afterwt = np.array(afterwt)
    row = [afterwt[0], afterwt[1]]
    # row = np.array(row)
    return row