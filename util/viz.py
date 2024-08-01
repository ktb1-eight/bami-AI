# viz.py

import folium
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def visualize_with_folium(locations: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray) -> folium.Map:
    """Visualize clustered data with Folium"""
    m = folium.Map(location=[36.5, 127.5], zoom_start=7)
    colors = list(mcolors.TABLEAU_COLORS.values())
    for idx, row in enumerate(locations):
        cluster_color = colors[labels[idx] % len(colors)]
        folium.CircleMarker(location=[row[1], row[0]], radius=3, color=cluster_color, fill=True, fill_color=cluster_color, fill_opacity=0.6).add_to(m)
    for center in cluster_centers:
        folium.Marker(location=[center[1], center[0]], icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    m.save('clustered_location_data_folium.html')
    return m

def visualize_with_plotly(locations: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray) -> None:
    """Visualize clustered data with Plotly Express"""
    df = pd.DataFrame(locations, columns=['Longitude', 'Latitude'])
    df['Cluster'] = labels
    fig = px.scatter_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        color='Cluster',
        zoom=5,
        mapbox_style="carto-positron",
        title='Clustered Location Data'
    )
    cluster_center_df = pd.DataFrame(cluster_centers, columns=['Longitude', 'Latitude'])
    fig.add_trace(
        go.Scattermapbox(
            lat=cluster_center_df['Latitude'],
            lon=cluster_center_df['Longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14,
                color='red',
                symbol='cross'
            ),
            showlegend=False
        )
    )
    fig.show()

def plot_silhouette(locations: np.ndarray, n_clusters: int=5, random_state: int=42) -> None:
    """Performs KMeans clustering and plots the silhouette coefficients."""

    # KMeans 군집화 수행
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    y_km = km.fit_predict(locations)

    # 군집 레이블과 군집 개수
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]

    # 실루엣 계수 계산
    silhouette_vals = silhouette_samples(locations, y_km, metric='euclidean')

    # 실루엣 플롯
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals,
                 height=1.0,
                 edgecolor='None',
                 color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    # 평균 실루엣 계수 플롯
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')

    # 플롯 레이블 설정
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()

    # 플롯 출력
    plt.show()

def add_markers_to_map(m: folium.Map, traveler_id: str, top_attractions: pd.DataFrame, color: str) -> None:
    for _, row in top_attractions.iterrows():
        lat, lon = row['Y_COORD'], row['X_COORD']
        attraction_name = row['VISIT_AREA_NM']
        
        if pd.notna(lat) and pd.notna(lon):
            tooltip = f'여행객: {traveler_id}, 어트랙션: {attraction_name}'
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color=color),
                tooltip=tooltip
            ).add_to(m)
        else:
            print(f"Skipping attraction {attraction_name} due to NaN values in coordinates.")
