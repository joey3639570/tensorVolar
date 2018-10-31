import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import datasets as ds
import concat_model
# Project Info
JobDir = 'C:\\Users\\User\\Documents\\Cheng Kung University\\2018 Big Data'
TrainingDataDir = os.path.join(JobDir, '0806_training_data')
# Dataset Info
LengthOfSeq = 7500
NumOfVars = 4
NumOfData = 40

now = datetime.datetime.now()
date = "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}".format(now.year, now.month, now.day, now.hour, now.minute)
MODEL = 'zscore-concat-fft-emd_5_avr5-wavelet_db4'
note = 'RMSE-fftset1-fft10-emd10_layer' #_avrPool100_frq400_preprocessing=log2_exp_std
MODE = 'eval'
LR = 0.005
LR_DECAY = 0.99
LR_DECAY_STEP = 1000
STEPS = 250000
# BASE_PATH = './train/{}/{}/{}'.format(MODEL, date, note)
BASE_PATH = 'train\{}\{}'.format(MODEL, note)
AUTO_EVAL = True
RESTORE_CHK_POINT = False
RESTORE_CHK_POINT_PATH = os.path.join(BASE_PATH, 'checkpoints', 'conv_model-250000')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'checkpoints')
SUMMARY_PATH = os.path.join(BASE_PATH, 'summary')
SAVE_CHK_POINT = True
SAVE_CHK_POINT_STEP = 5000
SAVE_SUMMARY = True


PLOT_PROCESS = False
if MODE == 'eval':
    RESTORE_CHK_POINT = True

def pretty_print(x):
    repr_str = np.array_repr(x)
    repr_str = repr_str.replace(' ','').replace('\n','')[7:-16]
    splits = np.array(repr_str.split(',')).astype(np.float32)
    template = '{:+.5f},\t'*(len(splits)-1) + '{:+5f}'
    formatted = template.format(*splits)
    return formatted

def main():
    if PLOT_PROCESS:
        plt.figure(figsize=[160, 4])
    with tf.device('/cpu:0'):
        """Loading datasets"""
        datasets = ds.Datasets(TrainingDataDir)
        o_train_data, o_quality, shapes = datasets.get_train_data(MODE)
        print('datasets.shape= ', o_train_data.shape)
        print('shapes= ', shapes)
        # print('datasets[0]= ', o_train_data[0])

        """Model Building"""
        train_data = tf.constant(o_train_data, shape=o_train_data.shape, dtype=tf.float32, name='train_data')
        quality = tf.constant(o_quality, shape=o_quality.shape, dtype=tf.float32, name='quality')

        fetches = concat_model.concat_train(inputs=train_data, lables=quality, shapes=shapes, mode=MODE, learning_rate=LR, save_summary=SAVE_SUMMARY)
        init_op = tf.global_variables_initializer()

        if SAVE_CHK_POINT or RESTORE_CHK_POINT:
            saver = tf.train.Saver(max_to_keep=10)
        if SAVE_SUMMARY and MODE != 'eval':
            summary_writer = tf.summary.FileWriter(SUMMARY_PATH, tf.get_default_graph())

        with tf.Session() as sess:
            if RESTORE_CHK_POINT:
                saver.restore(sess, RESTORE_CHK_POINT_PATH)
                print('restore variables from ', RESTORE_CHK_POINT_PATH)
            else:
                sess.run(init_op)
                print('Variables init')
            out = {}
            if MODE == 'train':
                for i in range(STEPS):
                    out = sess.run(fetches)
                    if (i + 1) % 500 == 0:
                        # print('step: {: >7},\t loss:{:.5E}, result:{:.5E}, quality:{:.5E}'.format(out['global_step'], out['ratial_loss'], out['result'], out['quality']))
                        print('step: {: >7},\t loss:{:.5E}'.format(out['global_step'], out['RMSE_loss']))
                        # print('step: {: >7}'.format(out['global_step']))
                        if SAVE_SUMMARY:
                            tf.summary.tensor_summary('RMSE_loss', fetches['RMSE_loss'])
                            tf.summary.tensor_summary('ratial_loss', fetches['ratial_loss'])
                            summary_writer.add_summary(out['summary_all'], global_step=out['global_step'])
                    if SAVE_CHK_POINT and (i + 1) % SAVE_CHK_POINT_STEP == 0:
                        saver.save(sess, CHECKPOINT_PATH + '/conv_model', global_step=out['global_step'])
            elif MODE == 'eval':
                out = sess.run(fetches)

            print('outputs:\n', pretty_print(out['result']))
            print('quality:\n', pretty_print(out['quality']))
            print('RMSE_loss:\n', np.sqrt(out['RMSE_loss']))
            print('ratial_loss:\n', np.sqrt(out['ratial_loss']))

            if AUTO_EVAL:
                pass


    if PLOT_PROCESS:
        plt.tight_layout()
        plt.savefig('test')
        plt.show()

if __name__ == '__main__':
    main()
