"""
Map Builder Module
Creates interactive Folium maps with pothole detection pins.
"""

import folium
from folium.plugins import HeatMap
from app.components.severity import get_severity_color, get_severity_emoji


def create_detection_map(detections: list[dict],
                          center_lat: float = 20.5937,
                          center_lon: float = 78.9629,
                          zoom: int = 5) -> folium.Map:
    """
    Create an interactive map with detection pins.

    Args:
        detections: List of detection dicts (must have latitude, longitude, severity)
        center_lat: Default map center latitude (India center)
        center_lon: Default map center longitude
        zoom: Default zoom level

    Returns:
        folium.Map object
    """
    # If we have GPS detections, center on the first one
    gps_detections = [
        d for d in detections
        if d.get("latitude") is not None and d.get("longitude") is not None
    ]

    if gps_detections:
        center_lat = gps_detections[0]["latitude"]
        center_lon = gps_detections[0]["longitude"]
        zoom = 15

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )

    # Add markers for each detection
    for det in gps_detections:
        severity = det.get("severity", "Unknown")
        confidence = det.get("confidence", 0)
        source = det.get("source_file", "Unknown")
        emoji = get_severity_emoji(severity)
        color_hex = get_severity_color(severity)

        # Map severity to Folium marker colors
        color_map = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
        }
        marker_color = color_map.get(severity, "blue")

        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; min-width: 180px;">
            <b>{emoji} Pothole Detected</b><br>
            <b>Severity:</b> {severity}<br>
            <b>Confidence:</b> {confidence:.0%}<br>
            <b>Source:</b> {source}<br>
            <b>Location:</b> {det['latitude']:.4f}, {det['longitude']:.4f}
        </div>
        """

        folium.Marker(
            location=[det["latitude"], det["longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{emoji} {severity} - {confidence:.0%}",
            icon=folium.Icon(color=marker_color, icon="exclamation-sign",
                             prefix="glyphicon")
        ).add_to(m)

    return m


def create_heatmap(detections: list[dict],
                    center_lat: float = 20.5937,
                    center_lon: float = 78.9629) -> folium.Map:
    """
    Create a heatmap showing pothole density.

    Args:
        detections: List of detection dicts with GPS coordinates
        center_lat: Default center latitude
        center_lon: Default center longitude

    Returns:
        folium.Map object with HeatMap layer
    """
    gps_detections = [
        d for d in detections
        if d.get("latitude") is not None and d.get("longitude") is not None
    ]

    if gps_detections:
        center_lat = gps_detections[0]["latitude"]
        center_lon = gps_detections[0]["longitude"]

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Prepare heat data: [lat, lon, weight]
    # Weight by severity: High=3, Medium=2, Low=1
    weight_map = {"High": 3, "Medium": 2, "Low": 1}
    heat_data = []

    for det in gps_detections:
        weight = weight_map.get(det.get("severity"), 1)
        heat_data.append([det["latitude"], det["longitude"], weight])

    if heat_data:
        HeatMap(
            heat_data,
            radius=20,
            blur=15,
            max_zoom=18,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 1.0: 'red'}
        ).add_to(m)

    return m


def create_mini_map(latitude: float, longitude: float,
                     severity: str = "High") -> folium.Map:
    """
    Create a small map with a single detection pin.
    Used for the image detection page.

    Args:
        latitude: Detection latitude
        longitude: Detection longitude
        severity: Severity level for pin color

    Returns:
        folium.Map object
    """
    color_map = {"Low": "green", "Medium": "orange", "High": "red"}
    marker_color = color_map.get(severity, "blue")

    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=16,
        tiles="OpenStreetMap",
        width="100%",
        height=300
    )

    folium.Marker(
        location=[latitude, longitude],
        tooltip=f"{get_severity_emoji(severity)} Pothole - {severity}",
        icon=folium.Icon(color=marker_color, icon="exclamation-sign",
                         prefix="glyphicon")
    ).add_to(m)

    return m