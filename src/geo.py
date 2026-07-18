"""Гео-расчёты, общие для EDA и feature engineering."""
import numpy as np

CENTER_LAT, CENTER_LON = 51.1283, 71.4304  # центр Астаны (район Байтерека)
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Векторизованное расстояние по формуле гаверсинусов, в километрах."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def distance_to_center(lat, lon):
    return haversine_km(lat, lon, CENTER_LAT, CENTER_LON)
