import preprocessing
import fftmodules
import hhtmodules
import waveletmodules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

class Datasets:
    def __init__(self, TrainingDataDir):
        self.LengthOfSeq = 7500
        self.NumOfVars = 4
        self.datasets = preprocessing.getSortedDataset(TrainingDataDir, self.LengthOfSeq, self.NumOfVars)
        # self.eval_index = [1, 5, 10, 15, 20, 25, 30, 35]
        self.eval_index = [7, 9, 13, 18, 34]
        self.train_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

    def get_NC_data(self, mode='train', zscore=False):
        """
        This function return the origin data in the form NC
        Number of batches, Channel, rows
        :return: List of NC data
        """
        assert mode in ['train', 'eval'], "Training mode should be 'train' or 'eval'"
        output = []
        quality = []
        sample_index = 0
        for sample in self.datasets:
            sample.dataList = np.transpose(a=sample.dataList, axes=[1, 0])
            if mode == 'train':
                if sample_index not in self.eval_index:
                    output.append(sample.dataList)
                    quality.append(sample.quality)
            else:
                if sample_index in self.eval_index:
                    output.append(sample.dataList)
                    quality.append(sample.quality)
            sample_index += 1
        output = np.array(output, dtype=np.float32)
        quality = np.array(quality, dtype=np.float32)
        print("Shape of output in datasets.py", output.shape)
        if zscore:
            output = stats.zscore(output)
        return output, quality

    def get_NHWC_data(self, mode='train', zscore=False):
        """
        This function return the origin data in the form NHWC
        Number of batches, Height, Width, Channel
        :return: List of NHWC data
        """
        assert mode in ['train', 'eval'], "Training mode should be 'train' or 'eval'"
        output = []
        quality = []
        sample_index = 0
        for sample in self.datasets:
            sample.dataList = np.transpose(a=sample.dataList, axes=[1, 0]) # 4*7500
            sample.dataList = np.reshape(sample.dataList, (4, 5, 1500))    # 4*5*1500
            sample.dataList = np.transpose(a=sample.dataList, axes=[1, 2, 0])  # 4*7500
            if mode == 'train':
                if sample_index not in self.eval_index:
                    output.append(sample.dataList)
                    quality.append(sample.quality)
            else:
                if sample_index in self.eval_index:
                    output.append(sample.dataList)
                    quality.append(sample.quality)
            sample_index += 1
        output = np.array(output, dtype=np.float32)
        quality = np.array(quality, dtype=np.float32)
        print("Shape of output in datasets.py", output.shape)
        if zscore:
            output = stats.zscore(output)
        return output, quality

    def get_NHWC_data_with_index(self, index=None, zscore=False):
        """
        This function return the origin data in the form NHWC according to the given index
        Number of batches, Height, Width, Channel
        :return: List of NHWC data
        """
        assert index, "Index must be given."
        output = []
        quality = []
        all_data = [data.dataList for data in self.datasets]
        global_min = np.min(a=all_data, axis=1, keepdims=True) # [40, 7500, 4]
        global_min = np.min(a=global_min, axis=0, keepdims=True)  # [40, 7500, 4]
        global_min = np.reshape(global_min, (4, 1))
        print('Global Min=', global_min)
        global_max = np.max(a=all_data, axis=1, keepdims=True)  # [40, 7500, 4]
        global_max = np.max(a=global_max, axis=0, keepdims=True)  # [40, 7500, 4]
        global_max = np.reshape(global_max, (4, 1))
        print('Global Min=', global_max)
        for sample_index in index:
            sample_dataList = self.datasets[sample_index].dataList
            sample_dataList = np.transpose(a=sample_dataList, axes=[1, 0])    # 4*7500
            if zscore:
                #minimum = np.min(a=sample_dataList, axis=1, keepdims=True)
                #maxmum = np.max(a=sample_dataList, axis=1, keepdims=True)
                minimum = global_min
                maxmum = global_max
                sample_dataList = (sample_dataList-minimum)/(maxmum-minimum)

            sample_dataList = np.reshape(sample_dataList, (4, 5, 1500))       # 4*5*1500
            sample_dataList = np.transpose(a=sample_dataList, axes=[1, 2, 0]) # 5*1500*4
            output.append(sample_dataList)
            quality.append(self.datasets[sample_index].quality)
        output = np.array(output, dtype=np.float32)
        quality = np.array(quality, dtype=np.float32)
        print("Shape of output in datasets.py", output.shape)
        # if zscore:
        #     output = stats.zscore(output)
        return output, quality

    def get_featured_data(self):
        """
        This function compute the features including fft, emd and wavelet, and pack them into a list with shape like:
        [batches, columns, algorithm, data]
        :return:    list of training data and quality
        """
        output = []
        quality = []
        for sample in self.datasets:
            union = []
            for varID in range(self.NumOfVars):
                y, ffty = fftmodules.fft_set1(sample.dataList[:, varID])        # 1*100
                # print('fft=', ffty.shape)
                emd = hhtmodules.emd_transformation(sample.dataList[:, varID])  # 5*75
                emd_out = np.hstack((emd[0], emd[1], emd[2]))                    # 1*225
                # print('emd= ', emd_out.shape)
                wavelet = waveletmodules.wavelet(sample.dataList[:, varID], waveletMode='db4')     # 2*6
                wavelet_out = np.hstack((wavelet[0], wavelet[1]))               # 1*12
                # print('wavelet= ', wavelet_out.shape)

                # ffty = tf.constant(ffty, dtype=tf.float32, name='ffty')
                # emd_out = tf.constant(emd_out, dtype=tf.float32, name='emd')
                # wavelet_out = tf.constant(wavelet_out, dtype=tf.float32, name='wavelet')
                ffty = np.array(ffty)
                emd_out = np.array(emd_out)
                wavelet_out = np.array(wavelet_out)

                # union.append([ffty, emd_out, wavelet_out])
                union = np.hstack([ffty, emd_out, wavelet_out])
            # union = np.array(union)
            output.append(union)
            quality.append(sample.quality)
        output = np.array(output)
        quality = np.array(quality)
        shapes = [ffty.shape[0], emd_out.shape[0], wavelet_out.shape[0]]
        return output, quality, shapes

    def get_train_data(self, mode='train', zscore=False):
        """
        Supply different combination of training data according to different mode.
        :param mode:    Only can br 'train' or 'eval'
        :return:        List of training data and quality
        """
        assert mode in ['train', 'eval'], "Training mode should be 'train' or 'eval'"
        o_datasets, o_target, shapes = self.get_featured_data()
        o_train_data = []
        o_quality = []
        for sample_index in range(len(o_datasets)):
            if mode == 'train':
                if sample_index not in self.eval_index:
                    o_train_data.append(o_datasets[sample_index])
                    o_quality.append(o_target[sample_index])
            else:
                if sample_index in self.eval_index:
                    o_train_data.append(o_datasets[sample_index])
                    o_quality.append(o_target[sample_index])
        o_train_data = np.array(o_train_data)
        o_quality = np.array(o_quality)
        if zscore:
            o_train_data = stats.zscore(o_train_data)
        return o_train_data, o_quality, shapes

    def plot_datasets(self):
        plt.subplot(8, 40 / cutter, int(80 * varID / cutter + sampleCount / cutter))
        plt.plot(fftx, y)
        # plt.axis([0, FRQ, 0, 0.00004])

        plt.subplot(8, 40 / cutter, int(80 * varID / cutter + sampleCount / cutter + 40 / cutter))
        plt.plot(freq, ffty)
        # plt.axis([0, FRQ, 0, 0.0005])

        """
        for sample in train_data:
            sampleCount += 1
            for varID in sample:
                if PLOT_PROCESS:
                    plt.subplot(3, 40, int(40 * 0 + sampleCount))
                    plt.plot(np.arange(len(varID[0])), varID[0])
                    plt.subplot(3, 40, int(40 * 1 + sampleCount))
                    plt.plot(np.arange(len(varID[1])), varID[1])
                    plt.subplot(3, 40, int(40 * 2 + sampleCount))
                    plt.plot(np.arange(len(varID[2])), varID[2])
        """
        '''
                    plt.subplot(8, 40/cutter, int(80*varID/cutter + sampleCount/cutter))
                    plt.plot(fftx, y)
                    #plt.axis([0, FRQ, 0, 0.00004])
    
                    plt.subplot(8, 40/cutter, int(80*varID/cutter + sampleCount/cutter+40/cutter))
                    plt.plot(freq, ffty)
                    #plt.axis([0, FRQ, 0, 0.0005])
        plt.tight_layout()
        # plt.axis([0, 7500, 0, 0.0002])200
        plt.savefig(FIG_NAME)
        plt.show()
        # plt.axis([0, 7500, -0.002, 0.002])
        '''