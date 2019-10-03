# -*- coding: utf-8 -*-
#реализовать алгоритм K-means:
import matplotlib.pyplot as plt
import copy
import numpy as np

x_mu0 = 30
y_mu0 = 50
x_sigma0 = 20
y_sigma0 = 40
x0 = x_mu0 + x_sigma0 * np.random.rand(100)
y0 = y_mu0 + y_sigma0 * np.random.rand(100)

x_mu1 = 60
y_mu1 = 70
x_sigma1 = 50
y_sigma1 = 10
x1 = x_mu1 + x_sigma1 * np.random.rand(100)
y1 = y_mu1 + y_sigma1 * np.random.rand(100)

x_mu2 = 50
y_mu2 = 80
x_sigma2 = 30
y_sigma2 = 40
x2 = x_mu2 + x_sigma2 * np.random.rand(100)
y2 = y_mu2 + y_sigma2 * np.random.rand(100)

x_mu3 = 10
y_mu3 = 20
x_sigma3 = 40
y_sigma3 = 20
x3 = x_mu3 + x_sigma3 * np.random.rand(100)
y3 = y_mu3 + y_sigma3 * np.random.rand(100)

x_mu4 = 20
y_mu4 = 40
x_sigma4 = 15
y_sigma4 = 70
x4 = x_mu4 + x_sigma4 * np.random.rand(100)
y4 = y_mu4 + y_sigma4 * np.random.rand(100)

x = np.concatenate((x0, x1, x2, x3, x4))
y = np.concatenate((y0, y1, y2, y3, y4))
data = np.array(list(zip(x, y)))


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def k_means():
    k = 5
    c_x = np.random.choice(x, size=k)
    c_y = np.random.choice(y, size=k)
    C = np.array(list(zip(c_x, c_y)))
    C_old = np.zeros(C.shape)
    labels = np.zeros(len(data))
    error = dist(C, C_old, None)
    while error != 0:
        for i in range(len(data)):
            distance = dist(data[i], C)
            cluster = np.argmin(distance)
            labels[i] = cluster
        C_old = copy.deepcopy(C)
        for i in range(k):
            points = [data[j] for j in range(len(data)) if labels[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
    plt.figure()
    plt.title("Кластеризация")
    colors = ['yellow', 'green', 'blue', 'red', 'purple']
    for i in range(k):
        points = np.array([data[j] for j in range(len(data)) if labels[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black') #конечный центроид
    plt.scatter(c_x, c_y, marker='*', s=200, c='orange') #начальный центроид
    plt.show()


k_means()

