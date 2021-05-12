import numpy as np


def Halving(K, m=1, lamb=10e-4, candidate_index=0):
    if candidate_index == 0:
        candidate_index = np.array(range(K.shape[0]))
    n = K.shape[0]
    m = min(n, m)
    q = len(candidate_index)
    index = np.zeros((m, 1))
    print('selecting samples ... ')
    for i in range(m):
        score = np.zeros((1, q))
        for j in range(q):
            k = candidate_index[j]
            score[0, j] = np.dot(K[k, :], (K[:, k])) / (K[k, k] + lamb)
        dummy = np.max(score)
        I = np.argmax(score)
        index[i] = candidate_index[I]
        K = K - ((K[:, int(index[i])]).reshape(-1, 1) * K[int(index[i]), :]) / (K[int(index[i]), int(index[i])] + lamb)
    print('done')
    index = index[0:m, 0]
    return index
