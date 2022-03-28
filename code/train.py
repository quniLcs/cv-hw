import tensorflow as tf
import numpy as np

from predict import predict
from backprop import backprop


# noinspection PyTypeChecker
def train(num_hidden, alpha, lambd):
    # Network Structure
    # Learning rate
    # Regularization parameter

    np.random.seed(0)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    del x_test, y_test
    x_train = np.reshape(x_train, (60000, 28 * 28))

    num_train = 60000
    num_label = 10
    dim = 28 * 28
    num_layer = len(num_hidden)

    iter_max = 100000
    iter_record = 20
    iter_step = iter_max // iter_record

    y_expanded = np.zeros((num_train, num_label))
    for i_train in range(num_train):
        y_expanded[i_train, y_train[i_train]] = 1

    weight = [np.random.rand(dim + 1, num_hidden[0])]
    for ind_layer in range(1, num_layer):
        weight.append(np.random.rand(num_hidden[ind_layer - 1] + 1, num_hidden[ind_layer]))
    weight.append(np.random.rand(num_hidden[num_layer - 1] + 1, num_label))

    error = [0 for _ in range(iter_record)]

    # cosine learning rate decay
    alpha = [alpha * np.cos(iter_cur / iter_max * np.pi / 2) for iter_cur in range(iter_max)]

    for iter_cur in range(iter_max):
        if iter_cur % iter_step == 0:
            ind_record = iter_cur // iter_step
            y_pred = predict(weight, x_train, num_layer)
            error[ind_record] = \
                sum([y_pred[i_train] != y_train[i_train] for i_train in range(num_train)]) / num_train
            print('Training iteration = %d\tTraining error = %f\n', iter_cur, error[ind_record])

        # Stochastic Gradient Descend
        index = int(np.random.rand() * num_train)
        grad = backprop()

        # L2 Regularization
        for ind_layer in range(num_layer + 1):
            weight[ind_layer] = weight[ind_layer] - \
                                alpha[iter_cur] * (grad + lambd * weight[ind_layer])


    # Model saving


if __name__ == "__main__":
    train([120, 84], 1e-3, 0.05)
