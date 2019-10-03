# -*- coding: utf-8 -*-
# минимизировать ошибку, подобрав наиболее лучший параметр модели, коэффициент регуляризации, базисную функцию
# отрисовать графики исходный и предсказанный
import numpy as np
from random import random, normalvariate, choice, sample

from math import sin, exp, pi, cos
import matplotlib.pyplot as plt

def generate_data(): #генерирование данных
    x_lst = [random() for _ in range(200)]
    x_lst.sort()
    y_lst = [20 * sin(2 * pi * 3 * x) + 100 * exp(x) for x in x_lst]
    t_lst = [y + normalvariate(0, 10) for y in y_lst]
    return x_lst, y_lst, t_lst


def split_data_train_test(x, y): #делит на train set, test set, validation set
    X_y = list(zip(x, y))
    np.random.shuffle(X_y)
    x, y = zip(*X_y)
    x = list(x)
    y = list(y)
    x_test = x[:int(len(x) * 0.8)]
    x_train = x[int(len(x) * 0.8):int(len(x) * 0.9)]
    x_valid = x[int(len(x) * 0.9):]
    y_test = y[:int(len(y) * 0.8)]
    y_train = y[int(len(y) * 0.8):int(len(y) * 0.9)]
    y_valid = y[int(len(y) * 0.9):]
    return x_test, x_train, x_valid, y_test, y_train, y_valid


x_lst, y_lst, t_lst = generate_data()

x_test, x_train, x_valid, y_test, y_train, y_valid = split_data_train_test(x_lst.copy(), y_lst.copy())


def get_phi(): #получение случайной базисной функции
    phi_c = [lambda x: cos(x), lambda x: sin(x), lambda x: np.sqrt(x),
             lambda x: x ** 2, lambda x: x ** 3, lambda x: x ** 4, lambda x: np.sqrt(x ** 3), lambda x: x ** 5,
             lambda x: x ** 6, lambda x: x ** 7,
             lambda x: x ** 8, lambda x: x ** 9, lambda x: x ** 10, lambda x: cos(x)]
    return sample(phi_c, 1)[0]


def get_random_lambda(): #получение случайного коэффициента регуляризации из списка
    lamdas_c = [0.0001, 0.1, 0.2, 0.001, 0.5, 0.8, 0.11, 0.011, 0.000000000001]
    return choice(lamdas_c)


def update_design_matrix(phi, x): #матрица плана
    design_matrix = np.zeros((len(x), 10))
    for i, k in enumerate(x):
        for j in range(10):
            if j == 0:
                design_matrix[i][j] = 1
            else:
                design_matrix[i][j] = phi(k**j)
    return design_matrix

def get_w(phi, x, y, lambda_): #вычисление параметра модели
    design_matrix = update_design_matrix(phi, x)
    a = design_matrix.T @ design_matrix
    b = lambda_ * np.eye(a.shape[0])
    c = a + b
    d = np.linalg.inv(c)
    e = d @ design_matrix.T
    w = e @ y
    return w


def predict_y(w, phi_curr, x, y): #
    d_matrix = update_design_matrix(phi_curr, x)
    y_model = np.dot(d_matrix, w)
    return y_model


def error_(w, phi_curr, x, y): #вычисление ошибки
    y_pred = predict_y(w, phi_curr, x, y)
    mse = np.mean((y - y_pred) ** 2)
    return mse, y_pred


def choose_params(): #
    Num_iters = 1000
    E_min = 1000000
    lambda_best = 0
    for i in range(Num_iters):
        phi_curr = get_phi()
        lamd_curr = get_random_lambda()
        w_curr = get_w(phi_curr, x_train, y_train, lamd_curr)
        E_curr, y_curr = error_(w_curr, phi_curr, x_valid, y_valid)
        if i == 0:
            w_best = w_curr
            phi_best = phi_curr
            E_model, y_model = error_(w_best, phi_best, x_test, y_test)
            E_min = E_model
            y_model = y_curr
        if E_curr < E_min:
            phi_best = phi_curr
            lambda_best = lamd_curr
            E_min = E_curr
            w_best = w_curr
            E_model, y_model = error_(w_best, phi_best, x_test, y_test)

    return lambda_best, y_model, phi_best



lambda_best, y_model, phi_best = choose_params()

#Generate new data and test

x_data, y_data, t_data = generate_data()
x_data.sort()

w_data = get_w(phi_best, x_data, y_data, lambda_best)
y_pred_data = predict_y(w_data, phi_best, x_data, y_data)
plt.figure()
plt.title("Регрессия")
plt.plot(x_data, y_pred_data, color="g")
plt.plot(x_data, y_data)
plt.plot(x_data, y_data, 'o')
plt.show()

