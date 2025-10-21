# **Geolocation Estimation** ![Version](https://img.shields.io/badge/Version-1.0.0-green)
## *Estimating Global Positioning Coordinates of Detected Objects using Computer Vision in Aerial Imagery*
<img src='preds/Prediction-Result.gif'></img>
Traditional methods to rapidly and accurately localize objects such as vehicles, pedestrians,
or animals over large areas often rely on hardware implants like GPS chips. These approaches are
typically costly, labor-intensive and limited to pre-selected targets. The proposed methodology
involves capturing high-resolution images or footage from Unmanned Aerial Vehicles (UAVs)
followed by state-of-the-art Computer Vision techniques to automate the detection and Geoloca-
tion of objects. The system framework leverages the UAV’s onboard sensor data, including gimbal
orientations, GPS positioning using GNSS-RTK technology and its altitude as primary parameters
for Geolocation Estimation. A primary challenge in estimating GPS coordinates from 2D images
is the lack of three-dimensional depth. In 2D images the ground is assumed to be flat, whereas in
real-world problems, most environments are non-euclidean, thus introducing error based on the
detected object’s altitude relative. To overcome this, our framework aims to incorporate Depth
Estimation and Digital Elevation Models (DEMs) for monocular cameras to significantly reduce
the error resulted by a non-euclidean ground for precise real-time object mapping. The primary
objective of this study is to develop a cost-effective and efficient object mapping system that is
expected to achieve high accuracy offering a scalable solution for applications in various domains
with the use of UAVs, such as wildlife monitoring, search and rescue, target tracking and more.


## 🚁 Overview

This system processes drone video footage to:
- Detect objects in real-time using YOLO models
- Convert pixel coordinates to GPS coordinates using drone metadata
- Generate a GIS map displaying the GPS coordinates of detected objects
- Create video outputs with geolocation overlays
- Export detection data for further analysis

## 🎯 Features

- **Real-time Object Detection**: Uses YOLOv8x trained on VisDrone dataset
- **Precise Geolocation**: Converts a single pixel point to GPS coordinates using drone camera parameters
- **Interactive GIS Visualization**: Live maps with satellite imagery and object tracking
- **Multi-format Output**: Video annotations, CSV data export, and live plotting
- **Drone Telemetry Integration**: Processes SRT subtitle files for flight data

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Dependencies
- Cuda Support: [**Install torch with Cuda**](https://pytorch.org/get-started/locally/)


## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Geolocation Estimation"
   ```
2. **Create miniconda Environment**
    ```
        conda create -n geolocation python==3.11 -y
        conda activate geolocation
    ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO model**
   - Place `yolov8x-visdrone.pt` in the `models/` directory or train your own using `train_model.py`


## 📁 Project Structure

```
├── data/                    # Geolocation Output data and labels
├── input/                   # Input video and SRT (Metadata) files
│   ├── DJI_VIDEO1.MP4
│   └── DJI_VIDEO1.SRT
├── models/                  # YOLO model files and other weights
├── object_detection/        # Detection modules
│   ├── inference.py
│   ├── load_model.py
│   └── draw_data.py
├── preds/                   # Prediction outputs and videos
├── src/                     # Core geolocation and GIS modules
│   ├── geolocation_estimation.py
│   └── gis_map.py
├── style/                   # UI assets, icons and more
├── utils/                   # Utility functions
│   ├── conversions.py
│   ├── image_utils.py
│   ├── load_srt.py
│   └── validation.py
├── run_geolocation.py       # Main execution script
└── train_model.py          # Model training script
```

## 🎮 Usage

### Basic Usage

1. **Prepare your input files**:
   - Place your drone video (`.MP4`) in the `input/` directory
   - Ensure you have the corresponding SRT subtitle file with the drone metadata
   - Update the `INPUT_VID` variable in `run_geolocation.py` to match your filename

2. **Run the geolocation estimation**:
   ```bash
   python run_geolocation.py
   ```

## 🎓 Model Training

To train custom models:

```bash
python train_model.py
```

The training script uses:
- **Dataset**: VisDrone format
- **Model**: YOLO11x architecture
- **Epochs**: 100 (configurable)
- **Image Size**: 640px

### Configuration Options

Edit the configuration variables in `run_geolocation.py`:

```python
# Input video file
INPUT_VID = "DJI_20240408125916_0001_W.MP4"
INPUT_DIR = f"input/{INPUT_VID}" # Input Directory

# Model and processing options
DET_MODEL_PATH = 'models/yolov8x-visdrone.pt'
OVERLAY_GIS = False         # Overlay GIS map on video
WRITE_GIS_VIDEO = True      # Generate separate GIS video
PLOT_LIVE = False           # Show live plotting window

# GIS map size
GIS_SIZE = (960, 640)
```

### Output Files

The system generates several output files:

- **`preds/[video_name]_preds.mp4`**: Annotated video with bounding boxes
- **`preds/gis_[video_name].mp4`**: GIS map visualization video
- **`data/stream_data.labels`**: CSV file with detection coordinates and metadata


## Supported Drones:
Current project version supports only **DJI M30/T**...


## 📊 Data Export

Detection data is exported in CSV format with columns:
- `lon`, `lat`: GPS coordinates
- `x`, `y`: Local coordinate system (meters)
- `cls`: Object class ID
- `cls+`: Object class name


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **VisDrone Dataset**: For object detection training data
- **Ultralytics**: For YOLO implementation
- **Folium**: For interactive mapping capabilities
- **OpenStreetMap**: For map tile services

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the code documentation

---

