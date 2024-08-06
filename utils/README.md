# Util Module

This folder contains various utility modules that are used throughout the project. These modules include functions for data processing, modeling, visualization, and general utilities.

## Folder Structure

util/<br>
├── data_processing.py<br>
├── modeling.py<br>
├── README.md<br>
├── utils.py<br>
└── viz.py


## Files and Functions

### `data_processing.py`

This file contains functions for processing and manipulating data.

#### Functions

- **get_season(date: pd.Timestamp) -> str**
  - Returns the season for a given date.
  - **Parameters:**
    - `date`: A pandas Timestamp object.
  - **Returns:** A string representing the season.

- **get_user_travel_style(style_mapping: List[str]) -> List[int]**
  - Collects user travel style preferences based on provided questions.
  - **Parameters:**
    - `style_mapping`: A list of questions to ask the user.
  - **Returns:** A list of integers representing the user's preferences.

- **calculate_similarity(users: pd.DataFrame, user_travel_style: List[int], vehicle_usage: str) -> List[Tuple[str, float]]**
  - Calculates similarity between users' travel styles.
  - **Parameters:**
    - `users`: A DataFrame containing users' travel styles.
    - `user_travel_style`: A list of the current user's travel style preferences.
    - `vehicle_usage`: A string indicating the user's vehicle usage.
  - **Returns:** A list of tuples containing traveler ID and similarity score.

- **find_nearest_attractions(atr_data: np.ndarray) -> np.ndarray**
  - Finds the nearest attractions starting from a given location.
  - **Parameters:**
    - `atr_data`: An array of attraction data (name, latitude, longitude).
  - **Returns:** An array of attractions sorted by proximity.

### `modeling.py`

This file contains functions related to machine learning modeling.

#### Functions

- **generate_kmeans_clusters(longitude: Union[pd.Series, np.ndarray], latitude: Union[pd.Series, np.ndarray], n_clusters: int=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]**
  - Generates KMeans clusters from longitude and latitude data.
  - **Parameters:**
    - `longitude`: Longitude data as a pandas Series or numpy array.
    - `latitude`: Latitude data as a pandas Series or numpy array.
    - `n_clusters`: The number of clusters.
  - **Returns:** A tuple containing locations, cluster centers, and labels.

- **compute_kmeans(k: int, locations: np.ndarray) -> Tuple[tuple, tuple]**
  - Computes inertia and silhouette score for a given number of clusters.
  - **Parameters:**
    - `k`: The number of clusters.
    - `locations`: An array of location data.
  - **Returns:** A tuple containing inertia and silhouette score.

- **find_cluster_idx(acc_lat: float, acc_lng: float, cluster_centers: np.ndarray) -> int**
  - Finds the cluster index for a given latitude and longitude.
  - **Parameters:**
    - `acc_lat`: Latitude of the point.
    - `acc_lng`: Longitude of the point.
    - `cluster_centers`: An array of cluster centers.
  - **Returns:** The index of the nearest cluster.

### `utils.py`

This file contains general utility functions that are commonly used across the project.

#### Functions

- **is_numeric(value: str) -> bool**
  - Checks if the given string value is numeric.
  - **Parameters:**
    - `value`: The string to check.
  - **Returns:** `True` if the value is numeric, otherwise `False`.

- **calculate_mse_similarity(user1: Union[list, pd.Series, pd.DataFrame], user2: Union[list, pd.Series, pd.DataFrame]) -> float**
  - Calculates the Mean Squared Error (MSE) similarity between two users.
  - **Parameters:**
    - `user1`: The first user's data.
    - `user2`: The second user's data.
  - **Returns:** The MSE similarity score.

### `viz.py`

This file contains functions for data visualization, primarily using Folium and Plotly.

#### Functions

- **visualize_with_folium(locations: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray) -> folium.Map**
  - Visualizes clustered data using Folium.
  - **Parameters:**
    - `locations`: Array of location data (longitude and latitude).
    - `cluster_centers`: Array of cluster center coordinates.
    - `labels`: Array of cluster labels.
  - **Returns:** A Folium map object.

- **visualize_with_plotly(locations: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray) -> None**
  - Visualizes clustered data using Plotly Express.
  - **Parameters:**
    - `locations`: Array of location data (longitude and latitude).
    - `cluster_centers`: Array of cluster center coordinates.
    - `labels`: Array of cluster labels.
  - **Returns:** None. Displays the Plotly map.

- **plot_silhouette(locations: np.ndarray, n_clusters: int=5, random_state: int=42) -> None**
  - Performs KMeans clustering and plots the silhouette coefficients.
  - **Parameters:**
    - `locations`: Array of location data.
    - `n_clusters`: Number of clusters.
    - `random_state`: Random state for reproducibility.
  - **Returns:** None. Displays the silhouette plot.

- **add_markers_to_map(m: folium.Map, traveler_id: str, top_attractions: pd.DataFrame, color: str) -> None**
  - Adds markers to a Folium map for a traveler's top attractions.
  - **Parameters:**
    - `m`: Folium map object.
    - `traveler_id`: Traveler's ID.
    - `top_attractions`: DataFrame containing top attractions with coordinates.
    - `color`: Color of the marker.
  - **Returns:** None.


Make sure to install all necessary dependencies before using the utility functions.

## Dependencies
This module requires the following packages:

- `pandas`
- `numpy`
- `folium`
- `plotly`
- `scikit-learn`
- `matplotlib`
- `autotime`
  
You can install these packages using pip:

```bash
pip install pandas numpy folium plotly scikit-learn matplotlib autotime
```

## Contributing
If you want to contribute to this module, please follow the guidelines below:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```go
이 `README.md` 파일은 `util` 폴더의 목적과 내용을 설명하며, 각 파일과 함수에 대한 간단한 설명을 제공합니다.
```