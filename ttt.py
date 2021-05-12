import numpy as np
import matplotlib.pyplot as plt
import scipy.io

data = 1 * np.random.randn(200, 2) + [7, 0]
data1 = 1 * np.random.randn(200, 2) + [7, 7]
data2 = 1.5 * np.random.randn(400, 2) + [0, 4]
label = np.zeros((200, 1))
label1 = np.ones((200, 1))
label2 = 2 * np.ones((400, 1))
data = np.vstack((data, data1))
data = np.vstack((data, data2))
label = np.vstack((label, label1))
label = np.vstack((label, label2))
# scipy.io.savemat('data.mat', {'data': data, 'label': label})
print(len(data))

plt.figure(figsize=(5, 5))
plt.subplot()
plt.scatter(data[:, 0], data[:, 1], c=label)
# plt.scatter(data1[:, 0], data1[:, 1], c='green')
# plt.scatter(data2[:, 0], data2[:, 1], c='red')
plt.show()
