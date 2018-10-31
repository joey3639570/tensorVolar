import tensorflow as tf
import numpy as np
import os
import PID_model
import readDataset as rd
import ProjectB_DatasetsTools as BatchGenerator
# Training Configuration
MODEL = 'PID'
note = '20181018'
MODE = 'eval'
LR = 0.0001
LR_DECAY = 0.99
LR_DECAY_STEP = 1000
BATCH_SIZE = 100
STEPS = 500000
# Model Configuration
DATA_LEN = 15
CONVERGENCE_STD = 0.01

BASE_PATH = os.path.join('..', 'train', MODEL, note) #'./train/{}/{}'.format(MODEL, note)
AUTO_EVAL = True
RESTORE_CHK_POINT = False
RESTORE_CHK_POINT_PATH = os.path.join(BASE_PATH, 'checkpoints', 'conv_model-35000')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'checkpoints')
SUMMARY_PATH = os.path.join(BASE_PATH, 'summary')
SAVE_CHK_POINT = True
SAVE_CHK_POINT_STEP = 5000
SAVE_SUMMARY = True

def main():
    """Loading datasets"""
    g14 = rd.Datasets('G14')
    miniG14 = rd.MiniDataset(g14)
    batches = BatchGenerator(miniG14, DATA_LEN, CONVERGENCE_STD)
    batches.shuffle()

    train_data = tf.placeholder(tf.float32, [DATA_LEN-1], name="input")
    real_output = tf.placeholder(tf.float32, [1], name="output")
    convergence_temperature = tf.constant(250, dtype=tf.float32, name='convergence_temperature')
    # Model
    fetches = PID_model.train(input=train_data,
                            label=real_output,
                            convgT=convergence_temperature,
                            dataLen=DATA_LEN,
                            mode=MODE,
                            learning_rate=LR,
                            learning_rate_decay=LR_DECAY,
                            save_summary=SAVE_SUMMARY)

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
                batch, label = batches.getBatch(BATCH_SIZE)
                out = sess.run(fetches, feed_dict={train_data: batch , real_output: label})
                if (i + 1) % 1000 == 0:
                    print('step: {: >7},\t loss:{:.5E}'.format(out['global_step'], out['loss']))
                    if SAVE_SUMMARY:
                        summary_writer.add_summary(out['summary_op'], global_step=out['global_step'])
                if SAVE_CHK_POINT and (i + 1) % SAVE_CHK_POINT_STEP == 0:
                    saver.save(sess, CHECKPOINT_PATH, global_step=out['global_step'])
        elif MODE == 'eval':
            pass
