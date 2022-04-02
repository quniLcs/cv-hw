import tensorflow as tf
import numpy as np
import pickle
import warnings
# with reference to:
# https://github.com/DWB1115
# https://blog.csdn.net/c976718017/article/details/79879496
# https://blog.csdn.net/jzz3933/article/details/84935205
# https://blog.csdn.net/jining11/article/details/81435899
# https://blog.csdn.net/graceful_snow/article/details/105187474
# https://blog.csdn.net/lly1122334/article/details/90647962

from predict import predict
from backprop import backprop


def train(num_hidden, alpha, lambd):
    # Network Structure
    # Learning rate
    # Regularization parameter

    np.random.seed(0)
    warnings.filterwarnings('ignore')

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)
    x_train = np.reshape(x_train, (60000, 28 * 28))

    mean = np.mean(x_train, axis = 0)
    std = [std if std > 1e-6 else 1 for std in np.std(x_train, axis = 0)]
    x_train = (x_train - mean) / std

    with open('mean.dat', 'wb') as file_pointer:
        pickle.dump(mean, file_pointer)
    with open('std.dat', 'wb') as file_pointer:
        pickle.dump(std, file_pointer)

    num_train = 60000
    num_label = 10
    dim = 28 * 28
    num_layer = len(num_hidden)

    iter_max = 100000
    iter_record = 20
    iter_step = iter_max // iter_record

    y_expanded = np.zeros((num_train, num_label))
    for i_train in range(num_train):
        y_expanded[i_train, y_train[i_train] % 10] = 1

    weight = [np.random.randn(dim + 1, num_hidden[0])]
    for ind_layer in range(1, num_layer):
        weight.append(np.random.randn(num_hidden[ind_layer - 1] + 1, num_hidden[ind_layer]))
    weight.append(np.random.randn(num_hidden[num_layer - 1] + 1, num_label))

    error = [0 for _ in range(iter_record)]

    # Cosine learning rate decay
    alpha = [alpha * np.cos(iter_cur / iter_max * np.pi / 2) for iter_cur in range(iter_max)]

    for iter_cur in range(iter_max):
        if iter_cur % iter_step == 0:
            ind_record = iter_cur // iter_step
            y_pred = predict(weight, x_train, num_layer)
            error[ind_record] = sum([y_pred[i_train] != y_train[i_train] for i_train in range(num_train)]) / num_train
            print('Training iteration = %d\tTraining error = %f' % (iter_cur, error[ind_record]))

        # Stochastic Gradient Descend
        ind_train = int(np.random.rand() * num_train)
        grad = backprop(weight,
                        x_train[ind_train: ind_train + 1, :],
                        y_expanded[ind_train: ind_train + 1, :],
                        num_layer, num_label)

        # L2 Regularization
        for ind_layer in range(num_layer + 1):
            weight[ind_layer] = weight[ind_layer] - alpha[iter_cur] * (grad[ind_layer] + lambd * weight[ind_layer])

    # Model saving
    file_name = 'model weights with num_hidden ' + str(num_hidden) + \
               ', alpha = ' + str(alpha[0]) + ', lambda = ' + str(lambd) + '.dat'
    with open(file_name, 'wb') as file_pointer:
        pickle.dump(weight, file_pointer)

    return error


if __name__ == "__main__":
    train([100], 1e-3, 0.05)
