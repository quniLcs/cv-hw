import numpy as np


def backprop(weight, x, y, num_layer):
    (num, dim) = x.shape

    grad = [np.zeros(weight[ind_layer].shape) for ind_layer in range(0, num_layer + 1)]

    for ind in range(num):
        net_activation = [np.matmul(np.hstack((x[ind, :], 1)), weight[0])]
        activation = [np.hstack((net_activation[0] * (net_activation[0] > 0), 1))]
        for ind_layer in range(1, num_layer):
            net_activation.append(np.matmul(activation[ind_layer - 1], weight[ind_layer]))
            activation.append(np.hstack((net_activation[ind_layer] * (net_activation[ind_layer] > 0), 1)))
        net_activation.append(np.matmul(activation[num_layer - 1], weight[num_layer]))
        # Cross entropy loss function
        softmax = np.exp(net_activation[num_layer]) / sum(np.exp(net_activation[num_layer]))

        # output layer
        error = softmax - y[ind, :]
        grad[num_layer] = grad[num_layer] + np.matmul(activation[num_layer - 1].T, error)
        error = (net_activation[num_layer - 1] > 0) * np.matmul(error, weight[num_layer][:, :-1].T)

        # hidden layers
        for ind_layer in range(num_layer - 1, -1, 1):
            grad[ind_layer] = grad[ind_layer] + np.matmul(activation[ind_layer - 1].T, error)
            error = (net_activation[ind_layer - 1] > 0) * np.matmul(error, weight[ind_layer][:-1, :-1].T)

        # input layer
        grad[0] = grad[0] + np.matmul(x[ind, :].T, error)
