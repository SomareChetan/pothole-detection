"""
Severity Classification Module
Classifies pothole severity based on bounding box area relative to frame area.
No separate ML model needed.
"""


# Thresholds (tunable)
LOW_THRESHOLD = 0.02       # < 2% of frame area
MEDIUM_THRESHOLD = 0.06    # 2% - 6% of frame area
                           # >= 6% is High


def classify_severity(bbox: tuple, image_width: int, image_height: int) -> str:
    """
    Classify pothole severity based on bounding box size relative to frame.

    Args:
        bbox: Normalized (x_center, y_center, width, height) where values are 0-1
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        "Low", "Medium", or "High"
    """
    _, _, bw, bh = bbox

    # Bounding box area as a ratio of total image area
    area_ratio = bw * bh

    if area_ratio < LOW_THRESHOLD:
        return "Low"
    elif area_ratio < MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"


def classify_severity_from_pixels(bbox_pixels: tuple,
                                   image_width: int,
                                   image_height: int) -> str:
    """
    Classify severity from pixel coordinates (x1, y1, x2, y2).

    Args:
        bbox_pixels: (x1, y1, x2, y2) pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        "Low", "Medium", or "High"
    """
    x1, y1, x2, y2 = bbox_pixels
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height

    area_ratio = bbox_area / image_area if image_area > 0 else 0

    if area_ratio < LOW_THRESHOLD:
        return "Low"
    elif area_ratio < MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"


def get_severity_color(severity: str) -> str:
    """Return hex color for severity level."""
    colors = {
        "Low": "#4CAF50",       # Green
        "Medium": "#FF9800",    # Orange
        "High": "#F44336",      # Red
    }
    return colors.get(severity, "#9E9E9E")


def get_severity_emoji(severity: str) -> str:
    """Return emoji for severity level."""
    emojis = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🔴",
    }
    return emojis.get(severity, "⚪")