from geolocation.geolocation_estimation2 import haversine_u, EQUATORIAL_RADIUS, get_meters_per_degree_lat
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import contextily as cx


def find_cluster_centers(file_path, dst_per_obj=3, _min_samples=20, class_column='cls'):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None
    print("Loaded Data...")

    if class_column not in data.columns:
        print(f"Error: Class column '{class_column}' not found in data.")
        print(f"Available columns: {data.columns.tolist()}")
        return None, None

    data['rad_lat'] = np.radians(data['lat'])
    data['rad_lon'] = np.radians(data['lon'])

    print("Performing DBSCAN...")
    dbscan = DBSCAN(eps=dst_per_obj/EQUATORIAL_RADIUS, min_samples=_min_samples, metric='haversine')
    print("Fitting...")
    clusters = dbscan.fit_predict(data[['rad_lat', 'rad_lon']])
    data['cluster'] = clusters
    print("Fitting Complete!...\nPreparing Data...")

    def get_mode(x):
        modes = x.mode()
        return modes.iloc[0] if not modes.empty else np.nan

    centers = (
        data[data['cluster'] != -1]
        .groupby('cluster')
        .agg(
            lon=('lon', 'mean'),
            lat=('lat', 'mean'),
            object_class=(class_column, get_mode)
        )
        .reset_index()
    )

    return data, centers


if __name__ == '__main__':
    print("Post-Process...")
    file_path = 'data/stream_data.labels'
    
    cls_id_to_name = {  
         0.0: "pedestrian",
         1.0: "people",
         2.0: "bicycle",
         3.0: "car",
         4.0: "van",
         5.0: "truck",
         6.0: "tricycle",
         7.0: "awning-tricycle",
         8.0: "bus",
         9.0: "motor"
    }

    cls_id_to_color = {
        0.0: "#1f77b4",
        1.0: "#ff7f0e",
        2.0: "#2ca02c",
        3.0: "#d62728",
        4.0: "#9467bd",
        5.0: "#8c564b",
        6.0: "#e377c2",
        7.0: "#7f7f7f",
        8.0: "#bcbd22",
        9.0: "#17becf"
    }
    
    data, centers = find_cluster_centers(file_path, dst_per_obj=0.7, _min_samples=20, class_column='cls')
    
    if centers is None or centers.empty:
        print("No data or clusters extracted... Exiting...")
        exit(-1)

    centers['class_name'] = centers['object_class'].map(cls_id_to_name)
    centers['color'] = centers['object_class'].map(cls_id_to_color)
    
    centers['class_name'] = centers['class_name'].fillna('Unknown')
    centers['color'] = centers['color'].fillna('#FFFFFF')
        
    print("Visualizing!")
    print(f"Num of Cluster Centers {len(centers)}")
    print("Cluster Centers (with classes):")
    print(centers) 
    
    fig, ax = plt.subplots()

    found_classes = centers['class_name'].unique()

    for cls_name in found_classes:
        class_data = centers[centers['class_name'] == cls_name]
        color = class_data['color'].iloc[0]
        
        ax.scatter(
            class_data['lon'], 
            class_data['lat'], 
            c=color, 
            s=50, 
            marker='.', 
            zorder=10, 
            label=cls_name
        )

    cx.add_basemap(ax, zoom=18, crs='EPSG:4326', source=cx.providers.Esri.WorldImagery)
    ax.set_title("Estimated GPS Clusters by Class")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    lon_min, lon_max = data['lon'].min(), data['lon'].max()
    lat_min, lat_max = data['lat'].min(), data['lat'].max()
    margin = 0.0008
    ax.set_xlim(lon_min - margin, lon_max + margin)
    ax.set_ylim(lat_min - margin, lat_max + margin)

    ax.legend()
    plt.show()