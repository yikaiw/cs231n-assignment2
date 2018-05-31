import numpy as np
import tensorflow as tf
import config


class Network(object):
    def __init__(self):
        self.X_inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])
        self.y_inputs = tf.placeholder(tf.int32, [None])
        self.labels = tf.one_hot(self.y_inputs, config.class_num, axis=1)
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.learning_rate, self.global_step, 2e3, 1e-4)
        
        self.logits = self.cnn(self.X_inputs)
        
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.accuracy = self.get_accuracy(self.logits, self.labels)
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    def cnn2(self, X_inputs):
        layer = tf.layers.conv2d(X_inputs, 32, 3, padding='same', activation=None)
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, self.keep_prob)
        layer = tf.layers.max_pooling2d(layer, 2, 2)
        
        layer = tf.layers.conv2d(X_inputs, 64, 3, padding='same', activation=None)
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, self.keep_prob)
        layer = tf.layers.max_pooling2d(layer, 2, 2)
        
        flat = tf.contrib.layers.flatten(layer)
        out = tf.layers.dense(flat, config.class_num)
        return out
    
    def cnn(self, X_inputs):
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs

            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(X_inputs, shape=[-1, 32, 32, 1])

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)
            
    def optimize(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(
            self.learning_rate, config.beta1, config.beta2).minimize(loss, global_step=self.global_step)
        return loss, optimizer

    def get_accuracy(self, logits, labels):
        softmax_logits = tf.nn.softmax(logits)
        correct_pre = tf.equal(tf.argmax(softmax_logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
        return accuracy
