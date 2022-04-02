import numpy as np


def backprop(weight, x, y, num_layer, num_label):
    (num, _) = x.shape

    grad = [np.zeros(weight[ind_layer].shape) for ind_layer in range(num_layer + 1)]

    for ind in range(num):
        # ReLU activation function
        net_activation = [np.matmul(np.hstack((x[ind: ind + 1, :], [[1]])), weight[0])]
        activation = [np.hstack((net_activation[0] * (net_activation[0] > 0), [[1]]))]
        for ind_layer in range(1, num_layer):
            net_activation.append(np.matmul(activation[ind_layer - 1], weight[ind_layer]))
            activation.append(np.hstack((net_activation[ind_layer] * (net_activation[ind_layer] > 0), [[1]])))
        net_activation.append(np.matmul(activation[num_layer - 1], weight[num_layer]))
        # Cross entropy loss function
        softmax = np.exp(net_activation[num_layer]) / np.sum(np.exp(net_activation[num_layer]))
        for ind_label in range(num_label):
            if np.isnan(softmax[0, ind_label]):
                break
        else:
            # output layer
            error = softmax - y[ind: ind + 1, :]
            grad[num_layer] += np.matmul(np.transpose(activation[num_layer - 1]), error)
            error = (net_activation[num_layer - 1] > 0) * np.matmul(error, np.transpose(weight[num_layer][:-1, :]))
            # hidden layers
            for ind_layer in range(num_layer - 1, 0, -1):
                grad[ind_layer] += np.matmul(np.transpose(activation[ind_layer - 1]), error)
                error = (net_activation[ind_layer - 1] > 0) * np.matmul(error, np.transpose(weight[ind_layer][:-1, :]))
            # input layer
            grad[0] += np.matmul(np.transpose(np.hstack((x[ind: ind + 1, :], [[1]]))), error)

    return grad
