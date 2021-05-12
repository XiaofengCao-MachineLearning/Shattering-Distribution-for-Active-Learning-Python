from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import NumberDensity
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from scipy.io import loadmat

# data = loadmat("./Syndata.mat")
# data = data['data']
# k = 4

# clf = KMeans(n_clusters=k)
# Id = clf.fit_predict(data)
# Cen = clf.cluster_centers_
def trains(data, Id):
    acc=[]
    for j in range(10, 101, 2):
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(data[0:j, :], Id[0:j])
        # a=clf.score(data[0:j, :], Id[0:j])
        lab = clf.predict(data)
        cnt = 0
        for i in range(len(Id)):
            if Id[i] != lab[i]:
                cnt += 1
        acc.append(cnt*1.0/len(data))
        # print(cnt * 1.0 / len(data))
    # print(a)
    plt.subplot(122)
    plt.plot(range(10,101,2),acc,'r')

