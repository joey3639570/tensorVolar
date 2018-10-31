import tensorflow as tf

def train(inputs, label, convgT, dataLen, mode='train', learning_rate=0.001, learning_rate_decay=0.99, save_summary=False):
    assert mode in ['train', 'eval'], "Training mode should be 'train' or 'eval'"
    
    global_step = tf.Variable(0, trainable=False)
    is_training = False
    if mode == 'train':
        is_training = True
    
    with tf.variable_scope('PID_model'):
        TE = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.float32, name='Temperature_Env')
        alpha = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.float32, name='Alpha')
        KI = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.float32, name='KI')
        KP = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.float32, name='KP')
        RF = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.float32, name='Rate_Forget')
        TN = inputs[-1]
        
        # Calculate intergral part
        eum = tf.range(dataLen-1, 1, -1)
        base = tf.ones(dataLen-1)
        base = base * RF
        base = tf.pow(RF, eum)
        INT = tf.reduce_sum(inputs * base)
        # Equation
        output = TN - alpha*(TN-TE) + ((convgT-TN)*(KP+KI)+KI*RF*INT) 
        loss = tf.square(label-output)
        if mode == 'train':
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=3000,
                                                        decay_rate=learning_rate_decay,
                                                        name='decayed_learning_rate',
                                                        staircase=True)
            if save_summary:
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('Temperature_Env', TE)
                tf.summary.scalar('Temperature_Now', TN)
                tf.summary.scalar('Alpha', alpha)
                tf.summary.scalar('KP', KP)
                tf.summary.scalar('KI', KI)
                tf.summary.scalar('Rate_Forget', RF)

                _activation_summaries(fetches)
                summary_op = tf.summary.merge_all()
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = opt.minimize(loss, global_step=global_step)
            fetches = {}
            fetches['loss'] = loss
            fetches['global_step'] = global_step
            fetches['train_op'] = train_op
    return fetches