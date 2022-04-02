import numpy as np


def predict(weight, x, y, num_layer, num_label):
    (num, _) = x.shape

    y_pred = []
    error = 0
    loss = 0

    for ind in range(num):
        # ReLU activation function
        net_activation = [np.matmul(np.hstack((x[ind, :], 1)), weight[0])]
        activation = [np.hstack((net_activation[0] * (net_activation[0] > 0), 1))]
        for ind_layer in range(1, num_layer):
            net_activation.append(np.matmul(activation[ind_layer - 1], weight[ind_layer]))
            activation.append(np.hstack((net_activation[ind_layer] * (net_activation[ind_layer] > 0), 1)))
        net_activation.append(np.matmul(activation[num_layer - 1], weight[num_layer]))
        y_pred.append(list(net_activation[num_layer]).index(max(net_activation[num_layer])))
        if y_pred[ind] != y[ind]:
            error += 1
        # Cross entropy loss function
        softmax = np.exp(net_activation[num_layer]) / np.sum(np.exp(net_activation[num_layer]))
        for ind_label in range(num_label):
            if np.isnan(softmax[ind_label]):
                break
        else:
            loss -= np.log(softmax[y[ind]] if softmax[y[ind]] > 1e-6 else 1e-6)

    return y_pred, error / num, loss / num
