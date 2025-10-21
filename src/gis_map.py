import folium
import numpy as np
import cv2
import time
import os
import tempfile

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

_DRIVER = None
_WINDOW_SIZE = None
_TEMP_HTML = os.path.join(tempfile.gettempdir(), 'folium_map.html')
_PERSISTENT_HTML = os.path.join(tempfile.gettempdir(), 'leaflet_persistent.html')
_PERSISTENT_READY = False


def _get_or_create_driver(width, height):
    global _DRIVER, _WINDOW_SIZE
    if _DRIVER is not None and _WINDOW_SIZE == (width, height):
        return _DRIVER

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument(f'--window-size={width},{height}')
    chrome_options.add_argument('--hide-scrollbars')
    chrome_options.add_argument('--disable-gpu')

    service = Service(ChromeDriverManager().install())
    _DRIVER = webdriver.Chrome(service=service, options=chrome_options)
    # Tighten page/script timeouts so stalls fail fast
    try:
        _DRIVER.set_page_load_timeout(6)
        _DRIVER.set_script_timeout(6)
    except Exception:
        pass
    _WINDOW_SIZE = (width, height)
    return _DRIVER


def _write_persistent_leaflet_html(zoom):
    # drone_icon_path = os.path.abspath("style/drone.png").replace("\\", "/")
    drone_pointer = os.path.abspath("style/Drone-Pointer.png").replace("\\", "/")
    drone_pointer_url = f"file:///{drone_pointer}"
    # drone_icon_url = f"file:///{drone_icon_path}"

    html = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
    <style>
        html, body, #map {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
        .drone-icon {{
            /* keep original positioning behavior */
            width: 100%;
            height: 100%;
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
            transform-origin: center center;
            image-rendering: -moz-crisp-edges;
            image-rendering: -webkit-crisp-edges;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
            filter: drop-shadow(0 0 6px rgba(0,255,0,0.9))
                    drop-shadow(0 0 3px rgba(0,0,0,0.7));
        }}
    </style>
    </head>
    <body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
    <script>
        (function() {{
        var zoom = {zoom};
        window._map = L.map('map', {{
            zoomControl: false,
            scrollWheelZoom: false,
            dragging: false,
            doubleClickZoom: false,
            touchZoom: false,
            minZoom: zoom,
            maxZoom: zoom
        }}).setView([0, 0], zoom);

        // Leaflet tile URL must keep single braces
        window._base = L.tileLayer('https://{{s}}.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}', {{
            maxZoom: zoom,
            subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
        }}).addTo(window._map);

        window._markers = L.layerGroup().addTo(window._map);
        window._overlays = L.layerGroup().addTo(window._map);
        window._droneLayer = L.layerGroup().addTo(window._map);

        window.updateMap = function(center, points, clss, poly, drone) {{
            try {{
            window._map.setView([center.lat, center.lon], window._map.getZoom(), {{ animate: false }});
            window._markers.clearLayers();
            window._overlays.clearLayers();
            window._droneLayer.clearLayers();

            for (let i = 0; i < points.length; i++) {{
                const p = points[i];
                const label = clss[i];
                const colorMap = {{
                                0: "#39FF14", // Electric Lime
                                1: "#1F51FF", // Vivid Blue
                                2: "#FF10F0", // Hot Pink
                                3: "#6F00FF", // Electric Indigo
                                4: "#FFB627", // Solar Flare Orange
                                5: "#A020F0", // Electric Purple
                                6: "#FF3131", // Screaming Red
                                7: "#DFFF00", // Chartreuse Yellow
                                8: "#00F5D4"  // Bright Teal
                                }};
                const color = colorMap[label] || "#7f7f7f";

                const icon = L.divIcon({{
                className: "custom-gis-marker-compact",
                html: `<div style="
                        display: flex;
                        align-items: center;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                        font-family: Arial, sans-serif;
                        font-size: 11px;
                        font-weight: bold;
                        color: white;">
                        <span style="
                            width: 12px;
                            height: 12px;
                            background-color: ${{color}};
                            border: 1px solid rgba(255,255,255,0.7);
                            border-radius: 50%;
                            margin-right: 4px;
                            box-shadow: 0 0 2px rgba(0,0,0,0.8);">
                        </span>
                        ${{label}}
                    </div>`,
                iconSize: null,
                iconAnchor: [5, 6]
                }});
                L.marker([p.lat, p.lon], {{ icon }}).addTo(window._markers);
            }}

            // 🎯 Multi-line target (unchanged)
            if (poly) {{
                Object.values(poly).forEach(line => {{
                L.polyline(line.map(p => [p[0], p[1]]), {{ color: '#7afbff', weight: 4 }}).addTo(window._overlays);
                }});
            }}

            if (drone) {{
                const yaw = (drone.yaw || 0) - 180;

                // dynamic size that changes with zoom but preserves positioning behavior
                const zoomLevel = window._map.getZoom();
                const baseSize = 50;
                const droneSize = Math.max(40, Math.round(baseSize * Math.pow(0.8, (zoom - zoomLevel))));

                const droneIcon = L.divIcon({{
                className: '',
                html: `<div class="drone-icon" style="
                            position: relative;
                            width: ${{droneSize}}px;
                            height: ${{droneSize}}px;
                            left: 50%;
                            top: 50%;
                            transform: translate(-50%, -50%) rotate(${{yaw}}deg);
                            background-image: url('{drone_pointer_url}');
                            background-repeat: no-repeat;
                            background-size: contain;
                            background-position: center;
                            image-rendering: crisp-edges;
                            filter: drop-shadow(0 0 6px rgba(0,255,0,0.9))
                                    drop-shadow(0 0 3px rgba(0,0,0,0.7));
                        "></div>`,
                iconSize: [droneSize, droneSize],
                iconAnchor: [droneSize / 2, droneSize / 2]
                }});
                L.marker([drone.lat, drone.lon], {{ icon: droneIcon }}).addTo(window._droneLayer);
            }}

            return true;
            }} catch(e) {{
            console.error(e);
            return false;
            }}
        }}

        window.mapReady = true;
        }})();
    </script>
    </body>
    </html>"""


    with open(_PERSISTENT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)


def _ensure_persistent_map(width, height, zoom=18):
    global _PERSISTENT_READY
    driver = _get_or_create_driver(width, height)
    if not _PERSISTENT_READY:
        _write_persistent_leaflet_html(zoom)
        driver.get(f'file:///{os.path.abspath(_PERSISTENT_HTML)}')
        WebDriverWait(driver, 8).until(
            lambda d: d.execute_script('return !!window.mapReady')
        )
        _PERSISTENT_READY = True
    return driver


def render_gis_persistent(coordinates, point_data, drone_data, poly_gps, shape, zoom=18):
    width, height = shape
    center_lat, center_lon, drone_yaw = drone_data['lat'], drone_data['lon'], drone_data['gimbal_yaw']
    driver = _ensure_persistent_map(width, height, zoom=zoom)
    clss = [str(int(_i['Type'])) for _i in point_data]
    center = { 'lat': float(center_lat), 'lon': float(center_lon) }

    drone_icon = {
        'lat': float(center_lat),
        'lon': float(center_lon),
        'yaw': float(drone_yaw),
        'url': 'style/drone-camera.png'
    }

    if coordinates is not None and len(coordinates) > 0:
        points = [{ 'lat': float(lat), 'lon': float(lon) } for (lat, lon) in coordinates]
    else:
        points = [{ 'lat': float(p['lat']), 'lon': float(p['lon']) } for p in (point_data or [])]
        
    poly = {k: [[float(p[0]), float(p[1])] for p in v] for k, v in poly_gps.items()}


    ok = driver.execute_script(
        "return window.updateMap(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);",
        center, points, clss, poly, drone_icon
    )

    
    time.sleep(0.25)

    screenshot = driver.get_screenshot_as_png()
    img_array = np.frombuffer(screenshot, dtype=np.uint8)
    img_bgr = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if img_bgr is None:
        raise ValueError('Failed to decode screenshot')
    if img_bgr.shape[1] != width or img_bgr.shape[0] != height:
        img_bgr = cv2.resize(img_bgr, (width, height))
    return img_bgr


def folium_map_to_numpy(folium_map, shape=(600, 400)):
    width, height = shape

    # Save the map to a persistent temp file to reuse across frames
    folium_map.save(_TEMP_HTML, close_file=False)

    try:
        driver = _get_or_create_driver(width, height)
        driver.get(f'file:///{os.path.abspath(_TEMP_HTML)}')

        # Wait for document ready (short, per-frame)
        WebDriverWait(driver, 8).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )

        # Ensure Leaflet recalculates layout after window-size hint
        try:
            driver.execute_script('window.dispatchEvent(new Event("resize"));')
        except Exception:
            pass

        def _tiles_loaded(d):
            return d.execute_script(
                "return (function(){\n"
                " var tiles = Array.from(document.querySelectorAll('.leaflet-tile'));\n"
                " var loading = document.querySelectorAll('.leaflet-tile-loading').length;\n"
                " if (tiles.length === 0) return false;\n"
                " var allOk = tiles.every(t => (t.complete !== false) && t.naturalWidth > 0);\n"
                " return allOk && loading === 0;\n"
                "})()"
            )

       
        try:
            WebDriverWait(driver, 8).until(_tiles_loaded)
        except Exception:
            # Retry once after a refresh
            driver.refresh()
            WebDriverWait(driver, 8).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            WebDriverWait(driver, 8).until(_tiles_loaded)

        # Small settle delay for rendering
        time.sleep(0.2)

        screenshot = driver.get_screenshot_as_png()
        img_array = np.frombuffer(screenshot, dtype=np.uint8)
        img_bgr = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        if img_bgr is None:
            raise ValueError("Failed to decode screenshot")

        if img_bgr.shape[1] != width or img_bgr.shape[0] != height:
            img_bgr = cv2.resize(img_bgr, (width, height))

        return img_bgr

    except Exception as e:
        print(f"Error converting folium map to numpy: {e}")
        fallback_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(fallback_img, "Map Error", (50, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return fallback_img

def create_gis_map_folium(coordinates, point_data, center_lat, center_lon, poly_gps, shape):
    width, height = shape
    
    m = folium.Map(location=[center_lat, center_lon], 
                   zoom_start=18,
                   width=width, height=height,
                   control_scale=False,
                   zoom_control=False,
                   tiles=None)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, Maxar, Earthstar Geographics',
        name='Esri Satellite',
        max_zoom=18,
        detect_retina=False,
        control=False
    ).add_to(m)

    # Lock zoom to imagery-supported level
    m.options['minZoom'] = 18
    m.options['maxZoom'] = 18
    
    
    # Add center marker
    folium.Marker(
        [center_lat, center_lon],
        icon=folium.Icon(color='green'),
        popup="Sakarya"
    ).add_to(m)
    
    # Add points with simplified styling
    for point in point_data:
        folium.CircleMarker(
            location=[point['lat'], point['lon']],
            radius=2,  # Smaller radius
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)
    
    # Only add essential lines
    top_left_gps, top_right_gps = poly_gps['top_left_gps'], poly_gps['top_right_gps']
    bot_left_gps, bot_right_gps = poly_gps['bot_left_gps'], poly_gps['bot_right_gps']
    
    # Add field of view lines
    folium.PolyLine(
        locations=[
            [center_lat, center_lon],
            [top_left_gps[0], top_left_gps[1]],
            [top_right_gps[0], top_right_gps[1]],
            [center_lat, center_lon]
        ],
        color='green',
        weight=2,
        fill=True,
        fill_color='green',
        fill_opacity=0.2
    ).add_to(m)
    
    # Add bottom line
    folium.PolyLine(
        locations=[
            [bot_left_gps[0], bot_left_gps[1]],
            [bot_right_gps[0], bot_right_gps[1]]
        ],
        color='red',
        weight=2
    ).add_to(m)
    
    # Convert to numpy array
    numpy_image = folium_map_to_numpy(m, shape)

    # Fallback: if imagery not available (placeholder tiles), re-render with OSM
    try:
        gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
        variance = float(gray.var())
        if variance < 20.0:
            m_osm = folium.Map(location=[center_lat, center_lon],
                               zoom_start=m.options.get('minZoom', 18),
                               width=width, height=height,
                               control_scale=False,
                               zoom_control=False,
                               tiles=None)
            folium.TileLayer(
                tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr='OpenStreetMap',
                min_zoom=m.options.get('minZoom', 18),
                max_zoom=m.options.get('maxZoom', 18),
                control=False
            ).add_to(m_osm)

            # Re-add overlays
            folium.Marker(
                [center_lat, center_lon],
                icon=folium.Icon(color='green'),
                popup="Sakarya"
            ).add_to(m_osm)
            for point in point_data:
                folium.CircleMarker(
                    location=[point['lat'], point['lon']],
                    radius=2,
                    color='red',
                    fill=True,
                    fill_color='red'
                ).add_to(m_osm)
            folium.PolyLine(
                locations=[
                    [center_lat, center_lon],
                    [top_left_gps[0], top_left_gps[1]],
                    [top_right_gps[0], top_right_gps[1]],
                    [center_lat, center_lon]
                ],
                color='green',
                weight=2,
                fill=True,
                fill_color='green',
                fill_opacity=0.2
            ).add_to(m_osm)
            folium.PolyLine(
                locations=[
                    [bot_left_gps[0], bot_left_gps[1]],
                    [bot_right_gps[0], bot_right_gps[1]]
                ],
                color='red',
                weight=2
            ).add_to(m_osm)

            numpy_image = folium_map_to_numpy(m_osm, shape)
    except Exception:
        pass

    return numpy_image