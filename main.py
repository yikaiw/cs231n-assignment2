import tensorflow as tf
import numpy as np
import math
import timeit
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from cnn import Network
from datetime import datetime
import os
import config
import platform
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint', None, 'Whether use a pre-trained checkpoint, default None.')

def main():
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    checkpoint_dir = 'checkpoints'
    if FLAGS.checkpoint is not None:
        checkpoint_path = os.path.join(checkpoint_dir, FLAGS.checkpoint.lstrip('checkpoints/'))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, '{}'.format(current_time))
        try:
            os.makedirs(checkpoint_path)
        except os.error:
            print('Unable to make checkpoints direction: %s' % checkpoint_path)
    model_save_path = os.path.join(checkpoint_path, 'model.ckpt')

    nn = Network()

    saver = tf.train.Saver()
    print('Build session.')
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    if FLAGS.checkpoint is not None:
        print('Restore from pre-trained model.')
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
        restore = tf.train.import_meta_graph(meta_graph_path)
        restore.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        step = int(meta_graph_path.split('-')[2].split('.')[0])
    else:
        print('Initialize.')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = 0

    loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    test_accuracy_list = []
    step = 0

    train_writer = tf.summary.FileWriter('logs/train@' + current_time, sess.graph)
    val_writer = tf.summary.FileWriter('logs/valid@' + current_time, sess.graph)
    summary_op = tf.summary.merge_all()

    print('Start training:')
    train_len = len(y_train)
    for epoch in range(config.num_epochs):
        permutation = np.random.permutation(train_len)
        X_train_data = X_train[permutation]
        y_train_data = y_train[permutation]
        data_idx = 0
        while data_idx < train_len - 1:
            X_train_batch = X_train_data[data_idx: np.clip(data_idx + config.batch_size, 0, train_len - 1)]
            y_train_batch = y_train_data[data_idx: np.clip(data_idx + config.batch_size, 0, train_len - 1)]
            data_idx += config.batch_size

            loss, _, train_accuracy, summary, lr = sess.run(
                [nn.loss, nn.optimizer, nn.accuracy, summary_op, nn.learning_rate],
                {nn.X_inputs: X_train_batch, nn.y_inputs: y_train_batch, nn.keep_prob: config.keep_prob, nn.training: True})
            loss_list.append(loss)
            train_accuracy_list.append(train_accuracy)
            print('>> At step %i: loss = %.2f, train accuracy = %.3f%%, learning rate = %.7f' 
                  % (step, loss, train_accuracy * 100, lr))
            train_writer.add_summary(summary, step)
            step += 1

        accuracy, summary = sess.run([nn.accuracy, summary_op],
            {nn.X_inputs: X_val, nn.y_inputs: y_val, nn.keep_prob: 1.0, nn.training: False})
        val_accuracy_list.append(accuracy)
        print('For epoch %i: valid accuracy = %.2f%%\n' % (epoch, accuracy * 100))
        val_writer.add_summary(summary, epoch)
        
    test_len = len(y_test)
    data_idx = 0
    while data_idx < test_len - 1:
        X_test_batch = X_test[data_idx: np.clip(data_idx + config.batch_size, 0, test_len - 1)]
        y_test_batch = y_test[data_idx: np.clip(data_idx + config.batch_size, 0, test_len - 1)]
        data_idx += config.batch_size

        test_accuracy = sess.run(nn.accuracy, 
            {nn.X_inputs: X_test_batch, nn.y_inputs: y_test_batch, nn.keep_prob: 1.0, nn.training: False})
        test_accuracy_list.append(test_accuracy)

        
    save_path = saver.save(sess, model_save_path, global_step=step)
    print('Model saved in file: %s' % save_path)
    sess.close()
    train_writer.close()
    val_writer.close()
    print('Test accuracy = %.2f%%\n' % (np.mean(test_accuracy_list) * 100))


if __name__ == '__main__':
    main()
