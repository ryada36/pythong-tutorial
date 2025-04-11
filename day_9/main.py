import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
points = np.random.rand(50, 2)

K=2
initial_indices = np.random.choice(points.shape[0], K, replace=False)
print(initial_indices)
centroids = points[initial_indices]
print(centroids)

def assign_clusters(points, centroids):
    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(points, labels, K):
    new_centroids = np.zeros((K, points.shape[1]))
    for k in range(K):
        new_centroids[k] = points[labels == k].mean(axis=0)
    return new_centroids

for i in range(10):  # 10 iterations for now
    labels = assign_clusters(points, centroids)
    centroids = update_centroids(points, labels, K)
    
    # Visualization
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x')
    plt.title(f"Iteration {i+1}")
    plt.show()