import tensorflow as tf

def concat_train(inputs, lables, shapes, mode='train', learning_rate=0.001, save_summary=False):
    pass

def concat_model(inputs, mode, shapes, save_summary=False):
    fft_size = shapes[0]
    emd_size = shapes[1]
    wavelet_size = shapes[2]
    column_size = fft_size + emd_size + wavelet_size
    assert mode in ['train', 'eval'], "Mode should be one of ['train', 'eval']"
    fetch = {}


def deal_with_fft_feature(inputs, fetch, shape, scope_name, save_summary):
    with tf.variable_scope(scope_name):
        """fft part"""