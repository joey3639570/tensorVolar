import os
import numpy as np
import matplotlib.pyplot as plt
import datasets as ds

# Project Info
JobDir = 'C:\\Users\\User\\Documents\\Cheng Kung University\\2018 Big Data'
TrainingDataDir = os.path.join(JobDir, '0806_training_data')

PLOT_PROCESS = True
FIG_NAME = 'zscore_emd'

def main():
    if PLOT_PROCESS:
        plt.figure(figsize=[160, 4])
    datasets = ds.Datasets(TrainingDataDir)
    train_data, quality, shapes = datasets.get_featured_data()

    fft_size = shapes[0]
    emd_size = shapes[1]
    wavelet_size = shapes[2]
    print('shapes=', shapes)
    column_size = fft_size + emd_size + wavelet_size

    sampleCount = 0
    for sample in train_data:
        sampleCount += 1
        if PLOT_PROCESS:
            plt.subplot(3, 40, int(40 * 0 + sampleCount))
            plt.plot(np.arange(len(sample[0:fft_size-1])), sample[0:fft_size-1])
            plt.axis([0, 100, -1, 5])

            plt.subplot(3, 40, int(40 * 1 + sampleCount))
            plt.plot(np.arange(len(sample[fft_size:fft_size+emd_size-1])), sample[fft_size:fft_size+emd_size-1])
            #plt.axis([0, , -0.3, 0.3])

            plt.subplot(3, 40, int(40 * 2 + sampleCount))
            plt.plot(np.arange(len(sample[fft_size+emd_size:column_size-1])), sample[fft_size+emd_size:column_size-1])

    plt.tight_layout()
    # plt.axis([0, 7500, 0, 0.0002])200
    plt.savefig(FIG_NAME)
    plt.show()
    # plt.axis([0, 7500, -0.002, 0.002])


if __name__ == '__main__':
    main()