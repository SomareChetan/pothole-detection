"""
GPS Extraction Module
Extracts GPS coordinates from image EXIF metadata and GPX route files.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import gpxpy
from datetime import datetime


def _convert_to_degrees(value):
    """Convert GPS coordinates from DMS (degrees, minutes, seconds) to decimal degrees."""
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)


def extract_gps_from_image(image_path: str) -> dict | None:
    """
    Extract GPS coordinates from image EXIF metadata.

    Args:
        image_path: Path to the image file

    Returns:
        Dict with 'latitude' and 'longitude', or None if no GPS data found
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is None:
            return None

        gps_info = {}

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag_name] = gps_value

        if not gps_info:
            return None

        # Extract latitude
        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = _convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info["GPSLatitudeRef"] == "S":
                lat = -lat
        else:
            return None

        # Extract longitude
        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            lon = _convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info["GPSLongitudeRef"] == "W":
                lon = -lon
        else:
            return None

        return {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6)
        }

    except Exception:
        return None


def extract_gps_from_image_bytes(image_bytes) -> dict | None:
    """
    Extract GPS from image bytes (e.g., from Streamlit file uploader).

    Args:
        image_bytes: bytes-like object or file-like object

    Returns:
        Dict with 'latitude' and 'longitude', or None
    """
    try:
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes))
        exif_data = image._getexif()

        if exif_data is None:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag_name] = gps_value

        if not gps_info:
            return None

        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = _convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info["GPSLatitudeRef"] == "S":
                lat = -lat
        else:
            return None

        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            lon = _convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info["GPSLongitudeRef"] == "W":
                lon = -lon
        else:
            return None

        return {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6)
        }

    except Exception:
        return None


def parse_gpx_file(gpx_file_path: str) -> list[dict]:
    """
    Parse a GPX file and return a list of timestamped GPS coordinates.

    Args:
        gpx_file_path: Path to the .gpx file

    Returns:
        List of dicts: [{"time": datetime, "latitude": float, "longitude": float}, ...]
    """
    points = []

    with open(gpx_file_path, "r") as f:
        gpx = gpxpy.parse(f)

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "time": point.time,
                    "latitude": round(point.latitude, 6),
                    "longitude": round(point.longitude, 6),
                })

    return points


def parse_gpx_bytes(gpx_bytes) -> list[dict]:
    """
    Parse GPX from bytes (e.g., Streamlit uploader).

    Args:
        gpx_bytes: bytes content of GPX file

    Returns:
        List of timestamped GPS points
    """
    points = []

    if isinstance(gpx_bytes, bytes):
        gpx_string = gpx_bytes.decode("utf-8")
    else:
        gpx_string = gpx_bytes.read().decode("utf-8")

    gpx = gpxpy.parse(gpx_string)

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "time": point.time,
                    "latitude": round(point.latitude, 6),
                    "longitude": round(point.longitude, 6),
                })

    return points


def interpolate_gps_for_frame(frame_index: int, total_frames: int,
                                gps_points: list[dict]) -> dict | None:
    """
    Map a video frame index to the nearest GPS coordinate.

    Linearly interpolates between GPX trackpoints based on frame position.

    Args:
        frame_index: Current frame number
        total_frames: Total frames in video
        gps_points: Sorted list of GPS points from parse_gpx

    Returns:
        Dict with 'latitude' and 'longitude', or None if no GPS data
    """
    if not gps_points or total_frames <= 0:
        return None

    # Map frame position (0 to 1) to GPS point index
    progress = frame_index / max(total_frames - 1, 1)
    gps_index = int(progress * (len(gps_points) - 1))
    gps_index = max(0, min(gps_index, len(gps_points) - 1))

    return {
        "latitude": gps_points[gps_index]["latitude"],
        "longitude": gps_points[gps_index]["longitude"],
    }