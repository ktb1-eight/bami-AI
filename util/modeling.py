# modeling.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Union

def generate_kmeans_clusters(longitude: Union[pd.Series, np.ndarray], latitude: Union[pd.Series, np.ndarray], n_clusters: int=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate KMeans clusters from longitude and latitude data."""
    # Ensure the input is a numpy array
    if isinstance(longitude, pd.Series):
        longitude = longitude.values
    if isinstance(latitude, pd.Series):
        latitude = latitude.values

    locations = np.column_stack((longitude, latitude))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(locations)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return locations, cluster_centers, labels

def compute_kmeans(k: int, locations: np.ndarray) -> Tuple[tuple, tuple]:
    """Function to compute inertia and silhouette score for a given k"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(locations)
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(locations, kmeans.labels_)
    return inertia, silhouette_avg

def find_cluster_idx(acc_lat: float, acc_lng: float, cluster_centers: np.ndarray) -> int:
    """Find the cluster index for a given latitude and longitude"""
    acc_loc = np.array([[acc_lat, acc_lng]])
    distances = np.linalg.norm(cluster_centers - acc_loc, axis=1)
    cluster_idx = np.argmin(distances)
    return cluster_idx + 1
