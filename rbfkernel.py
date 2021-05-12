import numpy as np


def rbfkernel(X, Y, sigma):
    N = X.shape[0]
    K = X.shape[1]
    M = Y.shape[0]

    Kxy = (np.ones((M, 1)) * np.sum((X ** 2).T, 0)).T + np.ones((N, 1)) * sum((Y ** 2).T, 1) - 2 * (np.dot(X,(Y.T)))
    Kxy = np.exp(-0.5 * Kxy / (sigma ** 2))
    return Kxy

