"""
Клиент для OpenStreetMap:

1. Поиск региона через Nominatim.
2. Загрузка сельскохозяйственных контуров через Overpass API.
"""

from typing import Any, Dict, List, Tuple

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": "vegetation-monitor/1.0 (contact@example.com)"
}


def search_regions(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Поиск региона по названию.

    Пример:
        search_regions("Краснодарский край")
    """
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": limit,
    }

    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()

    results = []

    for item in response.json():
        bbox_raw = item.get("boundingbox")

        if bbox_raw and len(bbox_raw) == 4:
            # Nominatim возвращает: [south, north, west, east]
            bbox = [float(x) for x in bbox_raw]
        else:
            bbox = None

        results.append(
            {
                "name": item.get("display_name"),
                "lat": float(item.get("lat")),
                "lon": float(item.get("lon")),
                "bbox": bbox,
            }
        )

    return results


def get_farmland_polygons(
    bbox: Tuple[float, float, float, float],
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Загрузка сельхозконтуров внутри bbox.

    Параметры:
        bbox = (south, west, north, east)

    Возвращает:
        GeoJSON FeatureCollection
    """
    south, west, north, east = bbox

    query = f"""
    [out:json][timeout:45];
    way
      ["landuse"~"farmland|cropland|orchard|vineyard|meadow|farm"]
      ({south},{west},{north},{east});
    out geom;
    """

    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers=HEADERS,
        timeout=80,
    )
    response.raise_for_status()

    data = response.json()

    features = []

    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue

        geometry = element.get("geometry")
        if not geometry:
            continue

        coordinates = [[point["lon"], point["lat"]] for point in geometry]

        if len(coordinates) < 3:
            continue

        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        tags = element.get("tags", {})

        feature = {
            "type": "Feature",
            "id": f"way/{element.get('id')}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates],
            },
            "properties": {
                "source": "openstreetmap",
                "osm_id": element.get("id"),
                "name": tags.get("name", ""),
                "landuse": tags.get("landuse", ""),
                "crop": tags.get("crop", ""),
            },
        }

        features.append(feature)

        if len(features) >= limit:
            break

    return {
        "type": "FeatureCollection",
        "features": features,
    }