import math
from typing import List, Tuple


class GeoUtils:
    @staticmethod
    def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Return great-circle distance in meters between two (lat, lon) points."""
        lat1, lon1 = a; lat2, lon2 = b
        rad = 6371008.8
        phi1 = math.radians(lat1); phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
        hav = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        central = 2 * math.asin(min(1, math.sqrt(hav)))
        return rad * central

    @staticmethod
    def path_length_km(coords: List[Tuple[float, float]]) -> float:
        """Return length of polyline (coords list) in kilometers."""
        if not coords or len(coords) < 2:
            return 0.0
        s = 0.0
        for i in range(1, len(coords)):
            s += GeoUtils.haversine_m(coords[i-1], coords[i])
        return s / 1000.0

    @staticmethod
    def densify_coords(coords: List[Tuple[float, float]], max_segment_m: int = 50) -> List[Tuple[float, float]]:
        """Insert intermediate points so that distance between points <= max_segment_m."""
        if len(coords) < 2:
            return coords[:]
        dense: List[Tuple[float, float]] = [coords[0]]
        for i in range(1, len(coords)):
            a = coords[i-1]; b = coords[i]
            d = GeoUtils.haversine_m(a, b)
            if d <= max_segment_m:
                dense.append(b)
                continue
            parts = int(math.ceil(d / max_segment_m))
            for k in range(1, parts+1):
                t = k / parts
                lat = a[0] + (b[0] - a[0]) * t
                lon = a[1] + (b[1] - a[1]) * t
                dense.append((lat, lon))
        return dense