from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

wine = load_wine()


#kNN
def KNN_Classifier():
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=7, p=3, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy of KNN n=7 on the training set: {:.3f}".format(knn.score(X_train, y_train)))
    print("Accuracy of KNN n=7 on the test set: {:.3f}".format(knn.score(X_test, y_test)))
    return X_test, y_pred


#Random Forest
def RandomForest_Classifier():
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    forest = RandomForestClassifier(n_estimators=100, random_state=11)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    print("Accuracy of Random Forest on the training set: {:.3f}".format(forest.score(X_train, y_train)))
    print("Accuracy of Random Forest on the test set: {:.3f}".format(forest.score(X_test, y_test)))
    return X_test, y_pred


X_test_knn, y_pred_knn = KNN_Classifier()
X_test_RF, y_pred_RF = RandomForest_Classifier()
colors = ['yellow', 'green', 'blue']
plt.figure()
plt.title("KNN")
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[5])
for i in range(0, 3):
    points = np.array([X_test_knn[j] for j in range(len(X_test_knn)) if y_pred_knn[j] == i])
    plt.scatter(points[:, 0], points[:, 5], c=colors[i], label=str(wine.target_names[i]))
plt.legend()
plt.figure()
plt.title("Random forest")
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[5])
for i in range(0, 3):
    points = np.array([X_test_RF[j] for j in range(len(X_test_RF)) if y_pred_RF[j] == i])
    plt.scatter(points[:, 0], points[:, 5], c=colors[i], label=str(wine.target_names[i]))
plt.legend()
plt.show()
