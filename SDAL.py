from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import NumberDensity
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import svmtt


def SDAL(data, k):
    # plt.figure(figsize=(10, 5))
    clf = KMeans(n_clusters=k)  #svmtest
    Id = clf.fit_predict(data)
    Cen = clf.cluster_centers_
    # svmtt.trains(data, Id)  #svmtest
    Center = Cen
    Radi = 0.25
    T = 0
    L = data.shape[0]
    R = data.shape[1]
    f = 0.
    f = NumberDensity.NumberDensity(data, Center, Radi)
    while T < 50:
        for j in range(k):
            Ball = []
            dist = []
            for i in range(L):
                dist.append(np.linalg.norm(data[i, :] - Center[j, :]))
                if dist[i] < Radi:
                    Ball.append(data[i, :])
            if len(Ball) == 0:
                Center[j, :] = Center[j, :]
            else:
                Center[j, :] = np.mean(np.array(Ball), 0)
        F = NumberDensity.NumberDensity(data, Center, Radi)

        cul = np.zeros((len(Center), len(Center)))
        flag = 0
        for j in range(len(Center)):
            for i in range(len(Center)):
                cul[i, j] = np.linalg.norm(Center[i, :] - Center[j, :])
                if i != j and cul[i, j] < 2 * Radi:
                    flag = 1
        # find = np.where(cul < 2 * Radi)[0]

        if F - f == 0 or flag:
            break
        else:
            f = F
        T += 1
        Radi = (1 + 0.1) * Radi
    # plt.subplot(121)  #svm test
    plt.scatter(data[:, 0], data[:, 1], c='blue')
    plt.scatter(Center[:, 0], Center[:, 1], c='red', marker='s')
    plt.scatter(Cen[:, 0], Cen[:, 1], c='green', marker='*')
    for center in Center:
        circle = plt.Circle((center[0], center[1]), Radi, color='red', fill=False)
        plt.gcf().gca().add_artist(circle)

    # clf = svm.SVC(decision_function_shape='ovo')
    # clf.fit(data, Id)
    # print(clf.predict(data))
    # print(Id)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(Center)
    Idx = neigh.kneighbors(Center)
    Center = np.zeros((len(Idx[1]), data.shape[1]))
    for i in range(len(Idx[1])):
        Center[i, :] = data[int(Idx[1][i]), :]
    return Center
