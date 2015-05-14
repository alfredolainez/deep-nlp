import numpy as np

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


