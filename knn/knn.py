from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
Y = iris.target
k = 5

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
n = X_train.shape[0]
m = X_test.shape[0]

# Евклидово расстояние
def distance(x1, x2): 
    return np.sqrt(np.sum((x1 - x2)**2))

# Ядро Епанечникова
def K(r):
    r = np.array(r)
    return np.where(np.abs(r) <= 1, 0.75 * (1 - r**2), 0)

# Предсказание одного объекта
def predict_one(x_test, X_train, Y_train, k):
    distances = np.array([distance(x_test, x) for x in X_train])
    sorted_indices = np.argsort(distances)

    nearest_indices = sorted_indices[:k]
    h_distance = distances[sorted_indices[k]]

    nearest_distances = distances[nearest_indices]
    r = nearest_distances / h_distance
    weights = K(r)
    neighbor_classes = Y_train[nearest_indices]

    class_weights = np.zeros(len(np.unique(Y_train)), dtype=float)
    for cls, w in zip(neighbor_classes, weights):
        class_weights[cls] += w

    return np.argmax(class_weights)

# predict для всего X_test
def predict(k):
    predictions = []
    for i in range(m):
        pred_class = predict_one(X_test[i], X_train, Y_train, k)
        predictions.append(pred_class)
    return [int(p) for p in predictions]

# LOO для всего X и Y
def LOO(X, Y, k):
    errors = 0
    n = X.shape[0]
    for i in range(n):
        X_train_loo = np.delete(X, i, axis=0)
        Y_train_loo = np.delete(Y, i)
        x_test = X[i]
        y_true = Y[i]

        y_pred = predict_one(x_test, X_train_loo, Y_train_loo, k)
        if y_pred != y_true:
            errors += 1

    return errors / n

# Поиск оптимального k и построение графика
def find_optimal_k(X, Y, max_k=20):
    loo_errors = []
    for ki in range(1, max_k + 1):
        err = LOO(X, Y, ki)
        loo_errors.append(err)

    optimal_k = np.argmin(loo_errors) + 1  # +1, потому что k начинается с 1
    optimal_error = loo_errors[optimal_k - 1]

    # accuracy на train/test
    predictions = predict(optimal_k)
    accuracy = np.mean(predictions == Y_test)

    # график
    plt.plot(range(1, max_k + 1), loo_errors, marker="o")
    plt.xlabel("k")
    plt.ylabel("LOO(k)")
    plt.title("LOO(k) для метода Парзеновского окна")
    plt.show()

    return optimal_k, optimal_error, accuracy


optimal_k, loo_value, acc = find_optimal_k(X, Y, max_k=20)
print("Оптимальное k:", optimal_k)
print("LOO(k):", loo_value)
print("Accuracy:", acc)

