"""
Given a list of distinct points (x, y), and the number of clusters k, cluster the points
into k clusters using k-means with random initialization.
"""
from random import randint
from collections import defaultdict
from sys import maxint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def initialize(points, k):
    """
    Given a list of points (x, y), returns a dictionary mapping each point to
    a random cluster between 1 and k.
    """
    cluster_assignment = {}
    for point in points:
        cluster_assignment[point] = randint(1, k)
    return cluster_assignment


def group_by_cluster(cluster_assignment):
    """
    Given a dictionary mapping each point to a cluster, return a dictionary
    mapping a cluster to a list of points in that cluster.
    """
    cluster_points = defaultdict(list)
    for point, cluster in cluster_assignment.items():
        cluster_points[cluster].append(point)
    return cluster_points


def compute_centroids(cluster_assignment):
    """
    Given a cluster assignment, compute the centroid for each cluster.
    A dictionary mapping the cluster number to the centroid is returned.
    """
    cluster_points = group_by_cluster(cluster_assignment)
    centroids = {}
    for cluster, points in cluster_points.items():
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        centroid = np.mean(x_values), np.mean(y_values)
        centroids[cluster] = centroid

    return centroids


def reassign_clusters(points, centroids):
    """
    Given a list of points and the centroid for each cluster, assign
    each point to the closest centroid.
    """
    cluster_assignment = {}
    for point in points:
        closest_centroid_cluster, closest_centroid_distance = None, maxint
        for cluster, centroid in centroids.items():
            distance = np.linalg.norm(np.array(point) - np.array(centroid))
            if distance < closest_centroid_distance:
                closest_centroid_cluster, closest_centroid_distance = cluster, distance
        cluster_assignment[point] = closest_centroid_cluster

    return cluster_assignment


def k_means(points, k, max_iterations=1000):
    """
    Perform k-means clustering until clusters no longer change or the max number of
    iterations has passed.
    """
    cluster_assignment = {}
    new_cluster_assignment = initialize(points, k)
    iterations = 0

    while new_cluster_assignment != cluster_assignment and iterations < max_iterations:
        cluster_assignment = new_cluster_assignment
        centroids = compute_centroids(cluster_assignment)
        new_cluster_assignment = reassign_clusters(points, centroids)
        iterations += 1

    return cluster_assignment


if __name__ == '__main__':
    points = [(0, 5), (1, 4), (2, 1), (4, 0), (5, 3)]
    k = 2
    cluster_assignment = k_means(points, k)
    colors = sns.color_palette("hls", k)
    print cluster_assignment
    cluster_points = group_by_cluster(cluster_assignment)
    for cluster, points in cluster_points.items():
        x = pd.Series([p[0] for p in points])
        y = pd.Series([p[1] for p in points])
        sns.regplot(x, y, color=colors[cluster-1], fit_reg=False)
    plt.show()

