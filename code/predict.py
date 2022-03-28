import numpy as np


def predict(weight, x, num_layer):
    (num, dim) = x.shape

    net_activation = []
    activation = []
    y_pred = []

    # ReLU activation function
    for ind in range(num):
        net_activation.append(np.matmul(np.hstack((x[ind, :], 1)), weight[0]))
        activation.append(np.hstack((net_activation[0] * (net_activation[0] > 0), 1)))
        for ind_layer in range(1, num_layer):
            net_activation.append(np.matmul(activation[ind_layer - 1], weight[ind_layer]))
            activation.append(np.hstack((net_activation[ind_layer] * (net_activation[ind_layer] > 0), 1)))
        net_activation.append(np.matmul(activation[num_layer - 1], weight[num_layer]))
        y_pred.append(list(net_activation[num_layer]).index(max(net_activation[num_layer])))

    return y_pred
