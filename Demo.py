import numpy as np
import rbfkernel
import SDAL
import Halving
from scipy.io import loadmat
import matplotlib.pyplot as plt

# getmat = loadmat("./data.mat")
getmat = loadmat("./Syndata.mat")
data = getmat['data']
# label = getmat['label']


K = rbfkernel.rbfkernel(data, data, 1.8)

ID = Halving.Halving(K, 400)

X = np.zeros((len(ID), data.shape[1]))
for i in range(ID.shape[0]):
    X[i, :] = data[int(ID[i]), :]

a = SDAL.SDAL(X, 4)
a = np.array(a)
plt.scatter(a[:, 0], a[:, 1], c='black', marker='x')
plt.show()
