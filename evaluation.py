import numpy as np
import theano.tensor as T

epsilon = 1.0e-15

def rmslog_loss(y_true, y_pred):
    return T.sqrt(T.sqr(T.log(1. + y_pred) - T.log(1. + y_true)).mean())

def rmslog_error(predicted, real):
    """
    Returns root mean squared logarithmic error

    This gives more importance to errors between 0 and 1 that errors between larger quantities such as 3 to 4.
    """
    error = 0
    for i in range(len(predicted)):
        error += (np.log(predicted[i] + 1) - np.log(real[i] + 1))**2
    error = np.sqrt(error / len(predicted))

    return error


