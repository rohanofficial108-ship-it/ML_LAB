import pandas as pd
import numpy as np
import random

# GenAI Tool Used: ChatGPT
# AI assistance was used for code generation.
# The K-Means algorithm is implemented without using
# sklearn KMeans so that it can be compared with the
# student's own implementation.


def euclidean_distance(point_a, point_b):

    total = 0

    for i in range(len(point_a)):
        total += (
            point_a[i] - point_b[i]
        ) ** 2

    return total ** 0.5


def assign_clusters(data, centroids):

    clusters = []

    for point in data:

        distances = []

        for centroid in centroids:

            distance = euclidean_distance(
                point,
                centroid
            )

            distances.append(distance)

        nearest_cluster = distances.index(
            min(distances)
        )

        clusters.append(nearest_cluster)

    return clusters


def calculate_centroids(
    data,
    clusters,
    k
):

    new_centroids = []

    for cluster_number in range(k):

        cluster_points = []

        for index in range(len(data)):

            if clusters[index] == cluster_number:

                cluster_points.append(
                    data[index]
                )

        if len(cluster_points) > 0:

            centroid = np.mean(
                cluster_points,
                axis=0
            )

        else:

            centroid = random.choice(data)

        new_centroids.append(centroid)

    return new_centroids


def kmeans(
    data,
    k,
    max_iterations=100
):

    centroids = [
        data[i].copy()
        for i in range(k)
    ]

    for iteration in range(
        max_iterations
    ):

        clusters = assign_clusters(
            data,
            centroids
        )

        new_centroids = calculate_centroids(
            data,
            clusters,
            k
        )

        if np.allclose(
            centroids,
            new_centroids
        ):

            break

        centroids = new_centroids

    return clusters, centroids, iteration + 1


def load_data():

    dataframe = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign"
    )

    numeric_dataframe = dataframe.select_dtypes(
        include=["int64", "float64"]
    )

    numeric_dataframe = numeric_dataframe.fillna(
        numeric_dataframe.mean()
    )

    return numeric_dataframe.values


def main():

    data = load_data()

    k = int(
        input(
            "Enter number of clusters: "
        )
    )

    clusters, centroids, iterations = kmeans(
        data,
        k
    )

    print("\nCluster Assignments:")
    print(clusters)

    print("\nFinal Centroids:")

    for index, centroid in enumerate(
        centroids
    ):

        print(
            "Centroid",
            index + 1
        )

        print(centroid)

    print(
        "\nNumber of iterations:",
        iterations
    )


if __name__ == "__main__":
    main()