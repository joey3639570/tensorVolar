import tensorflow as tf
import numpy as np

def concat_train(inputs, lables,shapes, mode='train', learning_rate=0.001, save_summary=False):
    """
    :param inputs:          the training data with shape [batches, columns, algorithms]
    :param lables:          the qualities of the input with shape [batches]
    :param mode:            train or eval
    :param save_summary:    whether to save summary
    :return:                the dictionary of variables and operation
    """
    global_step = tf.Variable(0, trainable=False)
    # inputs = tf.constant(inputs, dtype=tf.float32, name='inputs')
    # quality = tf.constant(lables, dtype=tf.float32, name='quality')

    fetches = concat_model(inputs=inputs, mode=mode, shapes=shapes, save_summary=save_summary)
    fetches['ratial_loss'] = ratial_loss(result=fetches['result'], target=lables)
    fetches['RMSE_loss'] = RMSE_loss(result=fetches['result'], target=lables)
    fetches['quality'] = lables
    # fetches['reduced_loss'] = tf.reduce_mean(fetches['ratial_loss'])

    if mode == 'train':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8)
        train_op = opt.minimize(fetches['RMSE_loss'], global_step=global_step)
        fetches['global_step'] = global_step
        fetches['train_op'] = train_op

    if save_summary:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.tensor_summary('ratial_loss', fetches['ratial_loss'])
        tf.summary.tensor_summary('quality', lables)
        fetches['summary_all'] = tf.summary.merge_all()
    return fetches

def deal_with_single_emd(inputs, startIndex, emd_size, name):
    net_emd = tf.layers.dense(inputs[:, startIndex:startIndex + emd_size - 1], units=50, name=name+'_1', activation=tf.nn.tanh,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
    net_emd = tf.layers.dense(net_emd, units=10, name=name+'_2', activation=tf.nn.tanh,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
    net_emd = tf.layers.dense(net_emd, units=1, name=name+'_3', activation=tf.nn.tanh,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
    net_emd = tf.squeeze(net_emd, axis=1)
    return net_emd


def deal_with_single_column(inputs, fetch, fft_size, emd_size, wavelet_size, scope_name, save_summary=False):
    with tf.variable_scope(scope_name):
        net_fft = tf.layers.dense(inputs[:, 0:fft_size-1], units=10, name='fft_dense1',activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        # net_fft = tf.layers.dense(net_fft, units=10, name='fft_dense2', activation=tf.nn.tanh,
        #                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_fft = tf.layers.dense(net_fft, units=4, name='fft_dense3',activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_fft = tf.layers.dense(net_fft, units=1, name='fft_dense4',activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_fft = tf.squeeze(net_fft, axis=1)
        fetch['{}_fft'.format(scope_name)] = net_fft

        # emd processing 1*225
        end_index = fft_size + emd_size
        startIndex = fft_size
        net_emd_1 = deal_with_single_emd(inputs=inputs, startIndex=startIndex, emd_size=emd_size//3, name='emd_dense1_1')
        startIndex += emd_size//3
        net_emd_2 = deal_with_single_emd(inputs=inputs, startIndex=startIndex, emd_size=emd_size//3, name='emd_dense1_2')
        startIndex += emd_size//3
        net_emd_3 = deal_with_single_emd(inputs=inputs, startIndex=startIndex, emd_size=emd_size//3, name='emd_dense1_3')
        print(emd_size//3)

        # net_emd = tf.layers.dense(inputs[:, fft_size:end_index-1], units=40, name='emd_dense1',activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_emd = tf.stack([net_emd_1, net_emd_2, net_emd_3], axis=1)
        net_emd = tf.layers.dense(net_emd, units=10, name='emd_dense2', activation=tf.nn.tanh,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_emd = tf.layers.dense(net_emd, units=5, name='emd_dense3', activation=tf.nn.tanh,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_emd = tf.layers.dense(net_emd, units=1, name='emd_dense4', activation=tf.nn.tanh,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_emd = tf.squeeze(net_emd, axis=1)
        fetch['{}_emd'.format(scope_name)] = net_emd

        # wavelet processing 1*12
        end_index += wavelet_size
        net_wavelet = tf.layers.dense(inputs[:, fft_size + emd_size:end_index-1], units=70, name='wavelet_dense1',activation=tf.nn.tanh,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_wavelet = tf.layers.dense(net_wavelet, units=50, name='wavelet_dens2',activation=tf.nn.tanh,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_wavelet = tf.layers.dense(net_wavelet, units=10, name='wavelet_dense3', activation=tf.nn.tanh,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_wavelet = tf.layers.dense(net_wavelet, units=1, name='wavelet_dense4', activation=tf.nn.tanh,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        net_wavelet = tf.squeeze(net_wavelet, axis=1)
        fetch['{}_wavelet'.format(scope_name)] = net_wavelet

        concat = tf.stack([net_fft, net_emd, net_wavelet], axis=1)
        col = tf.layers.dense(concat, units=1, name='concat', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
        col = tf.squeeze(input=col)
        fetch['{}'.format(scope_name)] = col
        if save_summary:
            tf.summary.tensor_summary('{}'.format(scope_name), col) # '{}'.format(scope_name)
        return fetch

def concat_model(inputs, mode, shapes, save_summary=False):
    """A simple fully connected model using concated features including fft,
    emd, wavelet.
    Input:  A tensor in shape [var, algorithm, data] (Only one sample at the same time)
    Output: Float number. The predicted quality."""

    fft_size = shapes[0]
    emd_size = shapes[1]
    wavelet_size = shapes[2]
    column_size = fft_size + emd_size + wavelet_size
    assert mode in ['train', 'eval'], "Mode should be one of ['train', 'eval']"
    fetch = {}
    """Column 0 Processing"""
    fetch = deal_with_single_column(inputs=inputs[:, 0:column_size-1], fetch=fetch, fft_size=fft_size
                                    , emd_size=emd_size, wavelet_size=wavelet_size, scope_name='column0', save_summary=save_summary)
    fetch = deal_with_single_column(inputs=inputs[:, column_size:2*column_size-1], fetch=fetch, fft_size=fft_size
                                    , emd_size=emd_size, wavelet_size=wavelet_size, scope_name='column1', save_summary=save_summary)
    fetch = deal_with_single_column(inputs=inputs[:, 2*column_size:3*column_size-1], fetch=fetch, fft_size=fft_size
                                    , emd_size=emd_size, wavelet_size=wavelet_size, scope_name='column2', save_summary=save_summary)
    fetch = deal_with_single_column(inputs=inputs[:, 3*column_size:4*column_size-1], fetch=fetch, fft_size=fft_size
                                    , emd_size=emd_size, wavelet_size=wavelet_size, scope_name='column3', save_summary=save_summary)
    concat = tf.stack([fetch['column0'], fetch['column1'], fetch['column2'], fetch['column3']], axis=1)
    result = tf.layers.dense(concat, units=1, name='result', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
    result = tf.squeeze(result, axis=1)
    fetch['result'] = result
    if save_summary:
        tf.summary.tensor_summary('result', result)
    return fetch

    """with tf.variable_scope('column0'):
        training_index = 0
        # fft processing 1*100
        print('aabbcc[0]= ', type(inputs[0][training_index]), inputs[0][training_index].shape)
        print(inputs[0][training_index])
        # print('aabbcc[1]= ', type(inputs[:][training_index][1]), inputs[:][training_index][1].shape)
        # print('aabbcc[2]= ', type(inputs[:][training_index][2]), inputs[:][training_index][2].shape)
        print('input= ', type(inputs), np.array(inputs).shape)

        aabbcc = tf.constant(np.array(inputs[:][training_index][0]), dtype=tf.float32)
        net_fft = tf.layers.dense(aabbcc, units=10, name='fft_dense1', activation=tf.nn.leaky_relu)
        net_fft = tf.layers.dense(net_fft, units=4, name='fft_dense2', activation=tf.nn.leaky_relu)
        net_fft = tf.squeeze(net_fft, axis=1)
        fetch['col0_fft'] = net_fft

        # emd processing 1*225
        net_emd = tf.layers.dense(inputs[:, training_index, 1], units=40, name='emd_dense1', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=10, name='emd_dense2', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=1, name='emd_dense3', activation=tf.nn.leaky_relu)
        net_emd = tf.squeeze(net_emd, axis=1)
        fetch['col0_emd'] = net_emd

        # wavelet processing 1*12
        net_wavelet = tf.layers.dense(inputs[:, training_index, 2], units=5, name='wavelet_dense1', activation=tf.nn.leaky_relu)
        net_wavelet = tf.layers.dense(net_wavelet, units=1, name='wavelet_dense2', activation=tf.nn.leaky_relu)
        net_wavelet = tf.squeeze(net_wavelet, axis=1)
        fetch['col0_wavelet'] = net_wavelet

        concat = tf.concat([net_fft, net_emd, net_wavelet], axis=1)
        col = tf.layers.dense(concat, units=1, name='concat')
        fetch['col0'] = col
        if save_summary:
            tf.summary.scalar('col0', col)

    ""Column 1 Processing""
    with tf.variable_scope('column1'):
        training_index = 1
        # fft processing 1*100
        net_fft = tf.layers.dense(inputs[:, training_index, 0], units=10, name='fft_dense1', activation=tf.nn.leaky_relu)
        net_fft = tf.layers.dense(net_fft, units=4, name='fft_dense2', activation=tf.nn.leaky_relu)
        net_fft = tf.squeeze(net_fft, axis=1)
        fetch['col1_fft'] = net_fft

        # emd processing 1*225
        net_emd = tf.layers.dense(inputs[:, training_index, 1], units=40, name='emd_dense1', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=10, name='emd_dense2', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=1, name='emd_dense3', activation=tf.nn.leaky_relu)
        net_emd = tf.squeeze(net_emd, axis=1)
        fetch['col1_emd'] = net_emd

        # wavelet processing 1*12
        net_wavelet = tf.layers.dense(inputs[:, training_index, 2], units=5, name='wavelet_dense1',
                                      activation=tf.nn.leaky_relu)
        net_wavelet = tf.layers.dense(net_wavelet, units=1, name='wavelet_dense2', activation=tf.nn.leaky_relu)
        net_wavelet = tf.squeeze(net_wavelet, axis=1)
        fetch['col1_wavelet'] = net_wavelet

        col = tf.layers.dense([net_fft, net_emd, net_wavelet], units=1, name='concat')
        fetch['col1'] = col
        if save_summary:
            tf.summary.scalar('col1', col)


    ""Column 2 Processing""
    with tf.variable_scope('column2'):
        training_index = 2
        # fft processing 1*100
        net_fft = tf.layers.dense(inputs[:, training_index, 0], units=10, name='fft_dense1', activation=tf.nn.leaky_relu)
        net_fft = tf.layers.dense(net_fft, units=4, name='fft_dense2', activation=tf.nn.leaky_relu)
        net_fft = tf.squeeze(net_fft, axis=1)
        fetch['col2_fft'] = net_fft

        # emd processing 1*225
        net_emd = tf.layers.dense(inputs[:, training_index, 1], units=40, name='emd_dense1', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=10, name='emd_dense2', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=1, name='emd_dense3', activation=tf.nn.leaky_relu)
        net_emd = tf.squeeze(net_emd, axis=1)
        fetch['col2_emd'] = net_emd

        # wavelet processing 1*12
        net_wavelet = tf.layers.dense(inputs[:, training_index, 2], units=5, name='wavelet_dense1',
                                      activation=tf.nn.leaky_relu)
        net_wavelet = tf.layers.dense(net_wavelet, units=1, name='wavelet_dense2', activation=tf.nn.leaky_relu)
        net_wavelet = tf.squeeze(net_wavelet, axis=1)
        fetch['col2_wavelet'] = net_wavelet

        concat = tf.concat([net_fft, net_emd, net_wavelet], axis=1)
        col = tf.layers.dense(concat, units=1, name='concat')
        fetch['col2'] = col
        if save_summary:
            tf.summary.scalar('col2', col)

    ""Column 3 Processing""
    with tf.variable_scope('column3'):
        training_index = 3
        # fft processing 1*100
        net_fft = tf.layers.dense(inputs[:, training_index, 0], units=10, name='fft_dense1', activation=tf.nn.leaky_relu)
        net_fft = tf.layers.dense(net_fft, units=4, name='fft_dense2', activation=tf.nn.leaky_relu)
        net_fft = tf.squeeze(net_fft, axis=1)
        fetch['col3_fft'] = net_fft

        # emd processing 1*225
        net_emd = tf.layers.dense(inputs[:, training_index, 1], units=40, name='emd_dense1', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=10, name='emd_dense2', activation=tf.nn.leaky_relu)
        net_emd = tf.layers.dense(net_emd, units=1, name='emd_dense3', activation=tf.nn.leaky_relu)
        net_emd = tf.squeeze(net_emd, axis=1)
        fetch['col3_emd'] = net_emd

        # wavelet processing 1*12
        net_wavelet = tf.layers.dense(inputs[:, training_index, 2], units=5, name='wavelet_dense1',
                                      activation=tf.nn.leaky_relu)
        net_wavelet = tf.layers.dense(net_wavelet, units=1, name='wavelet_dense2', activation=tf.nn.leaky_relu)
        net_wavelet = tf.squeeze(net_wavelet, axis=1)
        fetch['col3_wavelet'] = net_wavelet

        concat = tf.concat([net_fft, net_emd, net_wavelet], axis=1)
        col = tf.layers.dense(concat, units=1, name='concat')
        fetch['col3'] = col
        if save_summary:
            tf.summary.scalar('col3', col)

    concat = tf.concat([fetch['col0'], fetch['col1'], fetch['col2'], fetch['col3']], axis=1)
    result = tf.layers.dense(concat, units=1, name='result')
    if save_summary:
        tf.summary.scalar('result', result)
    return fetch"""

def ratial_loss(result, target):
    return tf.reduce_mean(tf.abs((result-target)/target))

def RMSE_loss(result, target):
    #result = tf.exp(result)
    #target = tf.exp(target)
    root = tf.losses.mean_squared_error(labels=target, predictions=result)
    root += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return root