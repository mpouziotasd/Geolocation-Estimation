
import numpy as np
import math

def get_intrinsics(drone_info):
    m30t_info = drone_info["M30T"]
    image_width = m30t_info["image_width"]
    image_height = m30t_info["image_height"]
    hfov_deg = m30t_info["horizontal_fov"]
    
    cx = image_width / 2.0
    cy = image_height / 2.0

    fx = cx / math.tan(math.radians(hfov_deg / 2.0))
    fy = fx

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

def get_rotation(drone_data):
    pitch_rad = math.radians(-drone_data['gimbal_pitch'])
    yaw_rad = math.radians(drone_data['gimbal_yaw'])
    
    R_yaw = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad), math.cos(yaw_rad), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    R_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad), math.cos(pitch_rad)]
    ], dtype=np.float64)

    R_matrix = R_yaw @ R_pitch
    return R_matrix

def get_translation(drone_data):
    east_m = 0.0
    north_m = 0.0
    up_m = drone_data['drone_alt']  # drone altitude AGL
    # Camera position in world coordinates (East, North, Up)
    return np.array([east_m, north_m, up_m], dtype=np.float64)