import os
import cv2 as cv
import csv

from object_detection.load_model import load_model
from utils.validation import VALIDATE_PROJECT_FILES
from utils.load_srt import get_srt_metadata
from utils.image_utils import overlay_gis
from object_detection.draw_data import draw_data
from src.gis_map import render_gis_persistent
from src.geolocation_estimation import pixels2gps, get_meters_per_degree_lat, get_drone_viewport
from pyproj import Proj
import contextily as cx

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

INPUT_VID = "DJI_20240408125916_0001_W.MP4"
INPUT_DIR = f"input/{INPUT_VID}"

DET_MODEL_PATH = 'models/yolov8x-visdrone.pt'
OVERLAY_GIS = True
WRITE_GIS_VIDEO = True
PLOT_LIVE = False


GIS_SIZE = (960, 640)
# GIS_SIZE = (1440, 960) # Larger GIS window

os.makedirs('input', exists_ok=True)

os.makedirs('preds', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

if not os.listdir('input'):
     print('WARNING: Empty input directory. Please add input drone videos and SRT files to the input/ folder.')
     exit(-1)
     
VALIDATE_PROJECT_FILES()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

drone_info = {"M30T":
                  {"vertical_fov": 44.1,
                   "horizontal_fov": 66,
                   "diagonal_fov": 79,
                   "focal_length_mm": 4.4,
                   "image_width": 3840,
                   "image_height": 2160,
                   "sensor_width": 6.4,
                   "sensor_height": 4.8,
                   "coc": 0.006
                   }
              }

cls_id = {  
     0: "pedestrian",
     1: "people",
     2: "bicycle",
     3: "car",
     4: "van",
     5: "truck",
     6: "tricycle",
     7: "awning-tricycle",
     8: "bus",
     9: "motor"}

with open('data/stream_data.labels', 'w', encoding='utf-8') as f:
     f.write('')
model = load_model('models/yolov8x-visdrone.pt')


source_vid = cv.VideoCapture(INPUT_DIR)
drone_metadata = get_srt_metadata(INPUT_DIR)

vid_name = INPUT_VID.replace('.MP4', '_preds.mp4')
out_vid_url = f"preds/{vid_name}"

fourcc = cv.VideoWriter_fourcc(*'XVID')
out_vid = cv.VideoWriter(out_vid_url, fourcc, 30, (1920, 1080))
if WRITE_GIS_VIDEO:
     gis_vid = cv.VideoWriter('preds/gis_'+vid_name, fourcc, 30, GIS_SIZE)

proj = Proj(proj='utm', zone=33, ellps='WGS84')
frame_idx = 0 
drone_path = []
stream_data = {'lon':[], 'lat':[], 'x':[], 'y':[], 'cls':[], 'cls+': []}
if PLOT_LIVE:
     plt.ion()
     fig, ax = plt.subplots()

lat0 = float(drone_metadata[0]['latitude'])
lon0 = float(drone_metadata[0]['longitude'])
drone_data = None
while source_vid.isOpened():
     ret, frame = source_vid.read()
     if not ret:
          source_vid.release()
          break
     if frame_idx < len(drone_metadata):
          # Updates drone data for the current frame if available
          drone_data = {
               "drone_alt": float(drone_metadata[frame_idx]['rel_alt']),
               "gimbal_pitch": float(drone_metadata[frame_idx]['gb_pitch']),
               "gimbal_yaw": float(drone_metadata[frame_idx]['gb_yaw']),
               "lat": float(drone_metadata[frame_idx]['latitude']),
               "lon": float(drone_metadata[frame_idx]['longitude']),
               'drone_info': drone_info
          }

     print(f"Processing frame {frame_idx}/{len(drone_metadata)}")
     results = model(frame, imgsz=1920, device=device, verbose=False)[0]

     xyxy = results.boxes.xyxy.cpu().numpy()
     clss = results.boxes.cls.cpu().numpy()

     det_coords = []

     current_lat = drone_data['lat']
     current_lon = drone_data['lon']

     shape = frame.shape[:2]
     drone_path.append((current_lat, current_lon))
     
     """
          Geolocation Estimation
     """
     polyline_coords = get_drone_viewport(drone_data, shape, drone_info)
     if len(xyxy) != 0:
          bboxes_np = np.array([box for box in xyxy])
          det_coords = pixels2gps(drone_data, bboxes_np, drone_info)

     gps_data = (
          [{"lon": float(lon), "lat": float(lat), "Type": i} for (lat, lon), i in zip(det_coords, clss)]
          if len(det_coords) != 0 else []
     )
     k = get_meters_per_degree_lat(lat0) # meters per degree latitude
     for (lat, lon), _cls in zip(det_coords, clss):
          x = (lat - lat0) * np.cos(np.radians(lat0)) * k # Meters
          y = (lon - lon0) * np.sin(np.radians(lon0)) * k
          stream_data['lon'].append(lon)
          stream_data['lat'].append(lat)
          stream_data['x'].append(x)
          stream_data['y'].append(y)
          stream_data['cls'].append(_cls)
          stream_data['cls+'].append(cls_id[_cls])

     
     if WRITE_GIS_VIDEO or OVERLAY_GIS:
          gis_img = render_gis_persistent(det_coords, gps_data,
                                        drone_data, polyline_coords, GIS_SIZE, zoom=18)

     processed_frame = draw_data(frame, xyxy, clss, cls_id) if xyxy.any() else frame
     processed_frame = cv.resize(processed_frame, (1920, 1080))
     if OVERLAY_GIS:
          processed_frame = overlay_gis(processed_frame, gis_img, GIS_SIZE)
     if WRITE_GIS_VIDEO:
          gis_vid.write(cv.cvtColor(gis_img, cv.COLOR_RGBA2BGR, gis_img))
     out_vid.write(processed_frame)
     
     frame_idx += 1
     lon = stream_data['lon']
     lat = stream_data['lat']
     if PLOT_LIVE:
          labels = [int(label) for label in stream_data['cls']]
          ax.clear()
          cmap = mpl.colormaps['viridis'].resampled(len(set(labels)))
          scatter = ax.scatter(lon, lat, c=labels, cmap=cmap, s=8, zorder=6)
          cx.add_basemap(ax, zoom=18, crs='EPSG:4326', source=cx.providers.Esri.WorldImagery)
          
          ax.set_title(f"GPS Tracking - Frame {frame_idx+1}")
          ax.set_xlabel("Longitude")
          ax.set_ylabel("Latitude")
          
          lon_range = np.max(lon) - np.min(lon)
          lat_range = np.max(lat) - np.min(lat)
          margin = 0.0008  # 5% of the max range
          ax.set_xlim(np.min(lon) - margin, np.max(lon) + margin)
          ax.set_ylim(np.min(lat) - margin, np.max(lat) + margin)
          unique_labels = sorted(set(labels))
          unique_int_labels = sorted(set(labels))
          handles = [Patch(color=cmap(i), label=stream_data['cls+'][j])
                    for i, j in enumerate([labels.index(lbl) for lbl in unique_int_labels])]

          ax.legend(handles=handles, title="Classes")
          plt.draw()
          plt.pause(0.1)

import pandas as pd

df = pd.DataFrame(stream_data)
df.to_csv('data/stream_data.labels', index=False, header=True)

print("Done")
if PLOT_LIVE:
     plt.clear()
     plt.ioff()
     plt.show()

