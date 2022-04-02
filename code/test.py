import tensorflow as tf
import numpy as np
import pickle

from predict import predict


def test(num_hidden, alpha, lambd):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    x_train = np.reshape(x_train, (60000, 28 * 28))
    x_test = np.reshape(x_test, (10000, 28 * 28))

    with open('mean.dat', 'rb') as file_pointer:
        mean = pickle.load(file_pointer)
    with open('std.dat', 'rb') as file_pointer:
        std = pickle.load(file_pointer)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    num_train = 60000
    num_test = 10000
    num_layer = len(num_hidden)

    # Model loading
    file_name = 'model weights with num_hidden ' + str(num_hidden) + \
               ', alpha = ' + str(alpha) + ', lambda = ' + str(lambd) + '.dat'
    with open(file_name, 'rb') as file_pointer:
        weight = pickle.load(file_pointer)

    # Training accuracy
    y_pred = predict(weight, x_train, num_layer)
    accuracy = sum([y_pred[i_train] == y_train[i_train] for i_train in range(num_train)]) / num_train
    print('Training accuracy = %f' % accuracy)
    # Testing accuracy
    y_pred = predict(weight, x_test, num_layer)
    accuracy = sum([y_pred[i_test] == y_test[i_test] for i_test in range(num_test)]) / num_test
    print('Testing accuracy = %f' % accuracy)


if __name__ == "__main__":
    test([100], 1e-3, 0.05)
