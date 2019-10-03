# -*- coding: utf-8 -*-
# Классифицирует спортсменов нв высоких и невысоких(баскетболистов и футболистов), исходя из заданного порога
# Изображает ROC-кривую и вычисляет AUC
import random
import matplotlib.pyplot as plt
from numpy import trapz
import numpy as np


n = 1000
T = 190
football = []
basketball = []

for i in range(0, n):
    football.append(random.normalvariate(170, 5))

for i in range(0, n):
    basketball.append(random.normalvariate(195, 7))


def predictions(threshold): #классифицирует по заданному порогу
    predictions_football = []
    predictions_basketball = []
    for i in range(0, n):
        if basketball[i] > threshold:
            predictions_basketball.append(1)
        else:
            predictions_basketball.append(0)
        if football[i] > threshold:
            predictions_football.append(1)
        else:
            predictions_football.append(0)
    return predictions_basketball, predictions_football


def metrics(threshold): #подсчет мертрик
    predictions_basketball, predictions_football = predictions(threshold)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(predictions_basketball)):
        if predictions_basketball[i] == 1:
            TP += 1
        else:
            FN += 1
        if predictions_football[i] == 0:
            TN += 1
        else:
            FP += 1
    return TP, TN, FP, FN


def count_errors(threshold): #подсчет accuracy, precision, recall, F1 score, alpha-error, beta-error
    TP, TN, FP, FN = metrics(threshold)
    if TP + FP == 0:
        precision = -1
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = -1
        beta = 0
    else:
        recall = TP / (TP + FN)
        beta = FN / (TP + FN)
    F1score = 2 * (precision * recall) / (precision + recall)
    if FP + TN == 0:
        alpha = -1
    else:
        alpha = FP / (FP + TN)
    accuracy = (TP + TN) / n
    false_positive_rate = 1 - TN / (TN + FP)
    true_positive_rate = recall
    return accuracy, precision, recall, F1score, alpha, beta, false_positive_rate, true_positive_rate


FPR = []
TPR = []
for i in range(0, T):
    accuracy, precision, recall, F1score, alpha, beta, false_positive_rate, true_positive_rate = count_errors(i)
    FPR.append(false_positive_rate)
    TPR.append(true_positive_rate)
print("FPR:")
print(FPR)
print("TPR:")
print(TPR)
area = trapz(TPR, dx=1)
print("Area under curve= " + str(area))
plt.figure()
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(FPR, TPR, 'r')
plt.show()






