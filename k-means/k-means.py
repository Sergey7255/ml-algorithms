from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
Y = iris.target
k = 3
n = X.shape[0]
m = 10  # количество инициализаций

# Евклидово расстояние
def distance(x1, x2): 
    return np.sqrt(np.sum((x1 - x2)**2))

def initialize_centers(X, k):
  
    np.random.seed()  # можно зафиксировать seed для повторяемости
    indices = np.random.choice(n, k, replace=False)
    centers = X[indices]
    return centers

centers = initialize_centers(X, k)

def assign_clusters(X, centers):
    labels = []
    for x in X:
        # вычисляем расстояния до всех центров
        dists = []
        for c in centers:
            d = distance(x, c)
            dists.append(d)
        # выбираем ближайший центр
        cluster = np.argmin(dists)
        labels.append(cluster)
    return np.array(labels)


def update_centers(X, clusters, k):
    n_features = X.shape[1]
    new_centers = np.zeros((k, n_features))
    
    for j in range(k):
        cluster_points = X[clusters == j]
        if len(cluster_points) > 0:
            new_centers[j] = cluster_points.mean(axis=0)
        else:
            # если кластер пустой, оставляем старый центр случайной точкой
            new_centers[j] = X[np.random.randint(0, X.shape[0])]
    
    return new_centers

def kmeans_one_run(X, k, max_iter=100, tol=1e-4):
    centers = initialize_centers(X, k)
    for i in range(max_iter):
        labels = assign_clusters(X, centers)   # E-шаг
        new_centers = update_centers(X, labels, k)  # M-шаг
        
        # проверяем, сильно ли изменились центры
        shift = np.sqrt(np.sum((centers - new_centers)**2))
        if shift < tol:
            break
        centers = new_centers
    
    return labels, centers

def clustering_quality(X, labels, centers):
    total = 0
    for j, center in enumerate(centers):
        cluster_points = X[labels == j]
        total += np.sum((cluster_points - center)**2)  # квадрат расстояния до центра
    return total

best_quality = np.inf
best_labels = None
best_centers = None

for i in range(m):
    labels, centers = kmeans_one_run(X, k)
    quality = clustering_quality(X, labels, centers)
    if quality < best_quality:
        best_quality = quality
        best_labels = labels
        best_centers = centers

plt.figure(figsize=(8, 6))

# Возьмем первые две координаты (два признака)
plt.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='viridis', s=40, alpha=0.7, label='Объекты')
plt.scatter(best_centers[:, 0], best_centers[:, 1], c='red', s=200, marker='X', label='Центры кластеров')

plt.title("Результат кластеризации методом K-Means")
plt.xlabel("Признак 1 (длина чашелистика)")
plt.ylabel("Признак 2 (ширина чашелистика)")
plt.legend()
plt.show()

print("Лучшие метки кластеров:\n", best_labels)
print("Лучшие центры:\n", best_centers)

























