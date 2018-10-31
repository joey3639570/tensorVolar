import tensorflow as tf
import re
slim = tf.contrib.slim
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
TOWER_NAME = 'tower'

def train(inputs, lables, mode='train', learning_rate=0.001, learning_rate_decay=0.99, stddev=1.0, save_summary=False):
    # Parameters for BatchNorm.

    assert mode in ['train', 'eval'], "Training mode should be 'train' or 'eval'"
    global_step = tf.Variable(0, trainable=False)
    is_training = False
    if mode == 'train':
        is_training = True
        inputs = tf.add(inputs, tf.truncated_normal(shape=inputs.shape, stddev=stddev, dtype=tf.float32))
    with tf.variable_scope('cnn_model'):
        # with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.tanh, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5)):
            result, fetches = model(inputs=inputs,
                                    dropout_keep_prob=0.8,
                                    is_training=is_training)

            fetches['ratial_loss'] = ratial_loss(result=fetches['result'], target=lables)
            fetches['RMSE_loss'] = RMSE_loss(result=fetches['result'], target=lables)
            fetches['quality'] = lables

            if mode == 'train':
                learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                           global_step=global_step,
                                                           decay_steps=3000,
                                                           decay_rate=learning_rate_decay,
                                                           name='decayed_learning_rate',
                                                           staircase=True)
                if save_summary:
                    # tf.summary.tensor_summary('RMSE_loss', fetches['RMSE_loss'])
                    # tf.summary.tensor_summary('ratial_loss', fetches['ratial_loss'])
                    tf.summary.scalar('learning_rate', learning_rate)
                    tf.summary.scalar('RMSE_loss', fetches['RMSE_loss'])
                    tf.summary.scalar('ratial_loss', fetches['ratial_loss'])

                    _activation_summaries(fetches)
                    summary_op = tf.summary.merge_all()
                    fetches['summary_op'] = summary_op
                #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                #with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = opt.minimize(fetches['RMSE_loss'], global_step=global_step)
                fetches['global_step'] = global_step
                fetches['train_op'] = train_op
    return fetches


def model(inputs, dropout_keep_prob=0.8, is_training=False):
    end_points = {}
    # input shape=[32, 5, 1500, 4]  [batches, height, width, rows]
    net = slim.max_pool2d(inputs, [1, 10], stride=[1, 10], padding='VALID', scope='first_max_pooling')
    print('Shape ', 'first_max_pooling', net.shape)
    tf.summary.image(name='first_pooling', tensor=net, max_outputs=32)
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        '''
        # input shape=[32, 5, 150, 4]  [batches, channels, height, width] NHWC
        end_point = 'Block_0'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 10, [5, 5], scope='Conv2d_0b_10_5x5')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 10, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_1 = slim.conv2d(branch_1, 10, [5, 1], scope='Conv2d_0b_10_5x1')
                branch_1 = slim.conv2d(branch_1, 10, [1, 3], scope='Conv2d_0c_10_1x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 10, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_2 = slim.conv2d(branch_2, 10, [1, 5], scope='Conv2d_0b_10_1x5')
                branch_2 = slim.conv2d(branch_2, 10, [1, 3], scope='Conv2d_0c_10_1x5')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.conv2d(net, 10, [1, 1], scope='Conv2d_0a_4_1x1')
            with tf.variable_scope('Branch_4'):
                branch_4 = slim.avg_pool2d(net, [1, 3], scope='Avr_pooling_0a_4_1x3')
                branch_4 = slim.conv2d(branch_4, 10, [1, 1], scope='Conv2d_0b_4_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3, branch_4])
        end_points[end_point] = net
        print('Shape ', end_point, net.shape)
        '''
        # input shape=[32, 5, 150, 50]  [batches, channels, height, width] NHWC
        end_point = 'Block_1'
        with tf.variable_scope(end_point):
            layer_deep = 2
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, layer_deep, [5, 5], scope='Conv2d_0b_10_5x5')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, layer_deep, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_1 = slim.conv2d(branch_1, layer_deep, [5, 1], scope='Conv2d_0b_10_5x1')
                branch_1 = slim.conv2d(branch_1, layer_deep, [1, 3], scope='Conv2d_0c_10_1x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, layer_deep, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_2 = slim.conv2d(branch_2, layer_deep, [1, 5], scope='Conv2d_0b_10_1x5')
                branch_2 = slim.conv2d(branch_2, layer_deep, [1, 3], scope='Conv2d_0c_10_1x5')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.conv2d(net, layer_deep, [1, 1], scope='Conv2d_0a_4_1x1')
            with tf.variable_scope('Branch_4'):
                branch_4 = slim.avg_pool2d(net, [1, 3], scope='Avr_pooling_0a_4_1x3')
                branch_4 = slim.conv2d(branch_4, layer_deep, [1, 1], scope='Conv2d_0b_4_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3, branch_4])
            # net = tf.layers.batch_normalization(inputs=net, training=is_training)
            # net = tf.nn.tanh(x=net)
        end_points[end_point] = net
        print('Shape ', end_point, net.shape)
        net_reduce = tf.reduce_mean(input_tensor=net, axis=3, keepdims=True)
        tf.summary.image(name=end_point, tensor=net_reduce, max_outputs=32)

        # input shape=[32, 5, 150, 20]  [batches, channels, height, width] NHWC
        end_point = 'Block_2'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                layer_deep = 4
                branch_0 = slim.conv2d(net, layer_deep, [5, 5], stride=[1, 2], scope='Conv2d_0b_10_5x5')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, layer_deep, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_1 = slim.conv2d(branch_1, layer_deep, [5, 1], scope='Conv2d_0b_10_5x1')
                branch_1 = slim.conv2d(branch_1, layer_deep, [1, 3], stride=[1, 2], scope='Conv2d_0c_10_1x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, layer_deep, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_2 = slim.conv2d(branch_2, layer_deep, [1, 5], scope='Conv2d_0b_10_1x5')
                branch_2 = slim.conv2d(branch_2, layer_deep, [1, 3], stride=[1, 2], scope='Conv2d_0c_10_1x5')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.conv2d(net, layer_deep, [1, 1], stride=[1, 2], scope='Conv2d_0a_4_1x1')
            with tf.variable_scope('Branch_4'):
                branch_4 = slim.avg_pool2d(net, [1, 3], scope='Avr_pooling_0a_4_1x3')
                branch_4 = slim.conv2d(branch_4, layer_deep, [1, 1], stride=[1, 2], scope='Conv2d_0b_4_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3, branch_4])
        end_points[end_point] = net
        print('Shape ', end_point, net.shape)
        net_reduce = tf.reduce_mean(input_tensor=net, axis=3, keepdims=True)
        tf.summary.image(name=end_point, tensor=net_reduce, max_outputs=32)

        # input shape=[32, 5, 75, 20]  [batches, channels, height, width] NHWC
        end_point = 'Block_3'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 4, [5, 5], stride=[1, 5], scope='Conv2d_0b_10_5x5')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 4, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_1 = slim.conv2d(branch_1, 4, [5, 1], scope='Conv2d_0b_10_5x1')
                branch_1 = slim.conv2d(branch_1, 4, [1, 5], stride=[1, 5], scope='Conv2d_0c_10_1x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 4, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_2 = slim.conv2d(branch_2, 4, [5, 5], scope='Conv2d_0b_10_1x5')
                branch_2 = slim.conv2d(branch_2, 4, [1, 5], stride=[1, 5], scope='Conv2d_0c_10_1x5')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.conv2d(net, 4, [1, 5], stride=[1, 5], scope='Conv2d_0a_4_1x1')
            with tf.variable_scope('Branch_4'):
                branch_4 = slim.avg_pool2d(net, [1, 5], scope='Avr_pooling_0a_4_1x3')
                branch_4 = slim.conv2d(branch_4, 4, [1, 1], stride=[1, 5], scope='Conv2d_0b_4_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3, branch_4])
        end_points[end_point] = net
        print('Shape ', end_point, net.shape)
        net_reduce = tf.reduce_mean(input_tensor=net, axis=3, keepdims=True)
        tf.summary.image(name=end_point, tensor=net_reduce, max_outputs=32)

        # input shape=[32, 5, 15, 20]  [batches, channels, height, width] NHWC
        end_point = 'Block_4'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 2, [5, 5], stride=[1, 3], scope='Conv2d_0b_10_5x5')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 2, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_1 = slim.conv2d(branch_1, 2, [5, 1], scope='Conv2d_0b_10_5x1')
                branch_1 = slim.conv2d(branch_1, 2, [1, 5], stride=[1, 3], scope='Conv2d_0c_10_1x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 2, [1, 1], scope='Conv2d_0a_10_1x1')
                branch_2 = slim.conv2d(branch_2, 2, [5, 5], scope='Conv2d_0b_10_1x5')
                branch_2 = slim.conv2d(branch_2, 2, [1, 5], stride=[1, 3], scope='Conv2d_0c_10_1x5')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.conv2d(net, 2, [1, 5], stride=[1, 3], scope='Conv2d_0a_4_1x1')
            with tf.variable_scope('Branch_4'):
                branch_4 = slim.avg_pool2d(net, [1, 5], scope='Avr_pooling_0a_4_1x3')
                branch_4 = slim.conv2d(branch_4, 2, [1, 1], stride=[1, 3], scope='Conv2d_0b_4_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3, branch_4])
        end_points[end_point] = net
        print('Shape ', end_point, net.shape)
        net_reduce = tf.reduce_mean(input_tensor=net, axis=3, keepdims=True)
        tf.summary.image(name=end_point, tensor=net_reduce, max_outputs=32)

        # input shape=[32, 5, 5, 10]  [batches, channels, height, width] NHWC
        with tf.variable_scope('Dense'):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net = slim.conv2d(net, 10, [5, 5], padding='VALID', scope='Conv2d_100_5x5')
                net = slim.conv2d(net, 5, [1, 1], padding='VALID', scope='Conv2d_50_1x1')
                end_points['Result_conv2d'] = net
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
                end_points['Dropout'] = net
                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                print('Shape ', 'Dropout', net.shape)
                result = tf.layers.dense(inputs=net, units=5, name='dense_1', activation=tf.nn.tanh,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
                result = tf.layers.dense(inputs=result, units=3, name='dense_2', activation=tf.nn.tanh,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
                result = tf.layers.dense(inputs=result, units=1, name='dense_3', activation=tf.nn.tanh,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=10e-5))
                result = tf.squeeze(result, axis=1, name='SpatialSqueeze')
                print('Shape ', 'Result', result.shape)
        end_points['result'] = result
    return result, end_points


def ratial_loss(result, target):
    return tf.reduce_mean(tf.abs((result-target)/target))#  + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))


def RMSE_loss(result, target):
    root = tf.losses.mean_squared_error(labels=target, predictions=result)
    root += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return root


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
    x: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.name)
    tensor_name = x
    tf.summary.histogram(name='{}/activations'.format(tensor_name), values=x)
    tf.summary.scalar(name='{}/sparsity'.format(tensor_name), tensor=tf.nn.zero_fraction(x))



def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)