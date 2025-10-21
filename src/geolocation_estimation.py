import math
import numpy as np

EQUATORIAL_RADIUS = 6378137.0  # meters
POLAR_RADIUS = 6356752.314245 
E2 = 1 - (POLAR_RADIUS**2 / EQUATORIAL_RADIUS**2)  # Squared Eccentricity


def get_meters_per_degree_lat(latitude):
    """Calculates meters per degree of latitude at a given latitude."""
    lat_rad = math.radians(latitude)

    m_num = EQUATORIAL_RADIUS * (1 - E2)
    m_den = (1 - E2 * math.sin(lat_rad) ** 2) ** 1.5
    m = m_num / m_den
    return (math.pi / 180) * m

def get_meters_per_degree_lon(latitude):
    """Calculates meters per degree of longitude at a given latitude."""
    lat_rad = math.radians(latitude)

    n = EQUATORIAL_RADIUS / math.sqrt(1 - E2 * math.sin(lat_rad) ** 2)
    return (math.pi / 180) * n * math.cos(lat_rad)

from math import radians, cos, sin, asin, sqrt
def haversine(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    R = 6371000  # Earth radius in meters
    
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c
def haversine_u(p1, p2):
    return haversine((p1[0], p1[1]), (p2[0], p2[1]))

def calc_dst2target(alt, nadir_angle_deg):
    """
        Trigonometric Calculation Desc:
            Calculating the base of a right triangle... 
    """

    nadir_angle_rad = np.radians(nadir_angle_deg)

    ground_dist = alt * np.tan(nadir_angle_rad)
    

    slant_dist = np.divide(alt, np.cos(nadir_angle_rad), 
                           out=np.full_like(nadir_angle_rad, np.inf), 
                           where=(np.cos(nadir_angle_rad) != 0))

    return ground_dist, slant_dist

def pixels2angles(coords, drone_info, gimbal_pitch_deg):
    """
    Converts pixel coordinates to angles using the camera's Field of View (FOV).
    """
    m30t_info = drone_info["M30T"]
    image_width = m30t_info["image_width"]
    image_height = m30t_info["image_height"]
    hfov_deg = m30t_info["horizontal_fov"]

    cx = image_width / 2.0
    cy = image_height / 2.0

    fx = cx / np.tan(np.radians(hfov_deg / 2.0))
    fy = fx  

    dx = (coords[:, 0] - cx) / fx
    dy = -(coords[:, 1] - cy) / fy
    

    rays = np.stack([dx, dy, np.ones_like(dx)], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    pitch_rad = np.radians(-gimbal_pitch_deg)
    
    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]],
                       dtype=np.float32)
    

    rays_rot = rays @ R_pitch.T
    
    dx_rot, dy_rot, dz_rot = rays_rot[:, 0], rays_rot[:, 1], rays_rot[:, 2]
    
    hor_ang_deg = np.degrees(np.arctan2(dx_rot, dz_rot))
    vert_ang_deg = np.degrees(np.arcsin(dy_rot))

    return hor_ang_deg, vert_ang_deg




def pixels2gps(drone_data, bboxes, drone_info):
    x_centers = (bboxes[:, 0] + bboxes[:, 2]) / 2
    y_centers = (bboxes[:, 1] + bboxes[:, 3]) / 2
    coords = np.column_stack((x_centers, y_centers))
    
    hor_ang, ver_ang = pixels2angles(coords, drone_info, drone_data['gimbal_pitch'])
    nadir_angle_deg = 90.0 + ver_ang
    
    ground_dsts, slant_dsts = calc_dst2target(drone_data['drone_alt'], nadir_angle_deg)
    
    geo_bearings_rads = np.radians((drone_data['gimbal_yaw'] + hor_ang) % 360)
    
    meters_per_degree_lat = get_meters_per_degree_lat(drone_data['lat'])
    meters_per_degree_lon = get_meters_per_degree_lon(drone_data['lat'])

    delta_north = ground_dsts * np.cos(geo_bearings_rads)
    delta_east = ground_dsts * np.sin(geo_bearings_rads)

    target_lats = drone_data['lat'] + (delta_north / meters_per_degree_lat)
    target_lons = drone_data['lon'] + (delta_east / meters_per_degree_lon)
    
    return np.column_stack((target_lats, target_lons))

def pixel2angles(coords, drone_info, gimbal_pitch_deg):
    m30t_info = drone_info["M30T"]
    image_width = m30t_info["image_width"]
    image_height = m30t_info["image_height"]
    hfov_deg = m30t_info["horizontal_fov"]

    cx = image_width / 2.0
    cy = image_height / 2.0

    fx = cx / math.tan(math.radians(hfov_deg / 2.0))
    fy = fx

    dx = (coords[0] - cx) / fx
    dy = -(coords[1] - cy) / fy

    ray = np.array([dx, dy, 1.0], dtype=np.float32)
    ray /= np.linalg.norm(ray)

    pitch_rad = math.radians(-gimbal_pitch_deg)
    R_pitch = np.array([[1, 0, 0],
                        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                        [0, math.sin(pitch_rad),  math.cos(pitch_rad)]], dtype=np.float32)

    ray_rot = R_pitch @ ray

    dx_rot, dy_rot, dz_rot = ray_rot

    hor_angle_deg = math.degrees(math.atan2(dx_rot, dz_rot))
    vert_angle_deg = math.degrees(math.asin(dy_rot))  # vertical angle from horizon

    return hor_angle_deg, vert_angle_deg

def pixel2gps(drone_data, pixel_point, drone_info):
    coords = np.array([pixel_point[0], pixel_point[1]])

    hor_ang, ver_ang = pixel2angles(coords, drone_info, drone_data['gimbal_pitch'])
    nadir_angle_deg = 90 + ver_ang

    ground_dst, slant_dst = calc_dst2target(drone_data['drone_alt'], nadir_angle_deg)

    geo_bearing_rad = np.radians((drone_data['gimbal_yaw'] + hor_ang) % 360)

    meters_per_degree_lat = get_meters_per_degree_lat(drone_data['lat'])
    meters_per_degree_lon = get_meters_per_degree_lon(drone_data['lat'])

    delta_north = ground_dst * np.cos(geo_bearing_rad)
    delta_east  = ground_dst * np.sin(geo_bearing_rad)

    target_lat = drone_data['lat'] + (delta_north / meters_per_degree_lat)
    target_lon = drone_data['lon'] + (delta_east / meters_per_degree_lon)

    return target_lat, target_lon

def pixels2dst(drone_data, image_shape, drone_info):
    """
    Compute slant distances for every pixel in the image.
    
    Args:
        drone_data: dict with drone info (altitude, gimbal_pitch, gimbal_yaw)
        image_shape: tuple (H, W)
        drone_info: dict with camera info
    
    Returns:
        ground_dsts: (H, W) array of ground distances in meters
        slant_dsts: (H, W) array of slant distances in meters
    """
    H, W = image_shape


    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    coords = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (H*W, 2)


    hor_ang, ver_ang = pixels2angles(coords, drone_info, drone_data['gimbal_pitch'])
    nadir_angle_deg = 90.0 + ver_ang

    ground_dsts, slant_dsts = calc_dst2target(drone_data['drone_alt'], nadir_angle_deg)

    # Reshape to image
    ground_dsts = ground_dsts.reshape(H, W)
    slant_dsts = slant_dsts.reshape(H, W)

    return ground_dsts, slant_dsts

    
def pixel2dst(drone_data, pixel_point, drone_info):
    
    coords = np.array([pixel_point[0], pixel_point[1]])

    hor_ang, ver_ang = pixel2angles(coords, drone_info, drone_data['gimbal_pitch'])
    nadir_angle_deg = 90 + ver_ang

    ground_dst, slant_dst = calc_dst2target(drone_data['drone_alt'], nadir_angle_deg)

    geo_bearing_rad = np.radians((drone_data['gimbal_yaw'] + hor_ang) % 360)

    delta_north = ground_dst * np.cos(geo_bearing_rad)
    delta_east = ground_dst * np.sin(geo_bearing_rad)
    return ground_dst, slant_dst, (delta_east, delta_north)

def get_drone_viewport(drone_data, shape, drone_info):
    """
        Creates the drone viewport using its telemetry data
    """
    height, width = shape
    x_inset = int(width * 0.2)
    y_inset = int(height * 0.2)

    top_left = [0, 0]
    top_right = [width - 1, 0]
    bot_left = [0, height - 1]
    bot_right = [width - 1, height - 1]

    x10 = int(width * 0.1)
    y10 = int(height * 0.1)
    x20 = int(width * 0.2)
    y20 = int(height * 0.2)

    coords = {
        'top_left_to_right': [
            pixel2gps(drone_data, top_left, drone_info),
            pixel2gps(drone_data, [x20, 0], drone_info)
        ],
        'top_left_to_bottom': [
            pixel2gps(drone_data, top_left, drone_info),
            pixel2gps(drone_data, [0, y20], drone_info)
        ],
        'top_right_to_left': [
            pixel2gps(drone_data, top_right, drone_info),
            pixel2gps(drone_data, [width - 1 - x20, 0], drone_info)
        ],
        'top_right_to_bottom': [
            pixel2gps(drone_data, top_right, drone_info),
            pixel2gps(drone_data, [width - 1, y20], drone_info)
        ],
        'bot_left_to_top': [
            pixel2gps(drone_data, bot_left, drone_info),
            pixel2gps(drone_data, [0, height - 1 - y20], drone_info)
        ],
        'bot_left_to_right': [
            pixel2gps(drone_data, bot_left, drone_info),
            pixel2gps(drone_data, [x20, height - 1], drone_info)
        ],
        'bot_right_to_top': [
            pixel2gps(drone_data, bot_right, drone_info),
            pixel2gps(drone_data, [width - 1, height - 1 - y20], drone_info)
        ],
        'bot_right_to_left': [
            pixel2gps(drone_data, bot_right, drone_info),
            pixel2gps(drone_data, [width - 1 - x20, height - 1], drone_info)
        ]
    }

    return coords
