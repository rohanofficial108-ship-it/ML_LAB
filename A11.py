import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("project_dataset.csv")
def kmeans(data, k, iterations):
    centroids = data[:k].copy()
    for _ in range(iterations):
        clusters = []
        for point in data:
            distances = []
            for centroid in centroids:
                distances.append(np.sqrt(np.sum((point-centroid)**2)))
            clusters.append(np.argmin(distances))
        new_centroids = []
        for i in range(k):
            cluster_points = data[np.array(clusters)==i]
            if len(cluster_points)>0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])
        centroids = np.array(new_centroids)
    return clusters, centroids
numeric_df = df.select_dtypes(include=["int64","float64"])
numeric_df = numeric_df.fillna(0)
data = numeric_df.values
clusters, centroids = kmeans(data, 3, 20)
print("Centroids")
print(centroids)
plt.scatter(data[:,0], data[:,1], c=clusters)
plt.scatter(centroids[:,0], centroids[:,1], marker="X", s=200)
plt.xlabel(numeric_df.columns[0])
plt.ylabel(numeric_df.columns[1])
plt.title("K-Means Clustering")
plt.show()