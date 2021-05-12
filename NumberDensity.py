import numpy as np


def NumberDensity(data, Center, Radius):
    f = 0.
    for i in range(len(data)):
        Ball_dist = []
        dist = []
        for j in range(len(Center)):
            dist.append(np.linalg.norm(data[i, :] - Center[j, :]))
            if dist[j] < Radius:
                a=np.array(dist[j])
                Ball_dist.append(dist[j])
        f = f + sum(np.exp(np.array(Ball_dist) / 1.8) ** 2) / (len(Ball_dist) + 1)
    return f
