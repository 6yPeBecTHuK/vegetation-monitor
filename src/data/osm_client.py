"""
Клиент OSM:
- поиск региона: Nominatim + Photon (фолбэк);
- сельхозконтуры: Overpass API (несколько зеркал).
"""

from typing import Any, Dict, List, Tuple

import requests
import concurrent.futures

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"

# Overpass-зеркала: первые три — основные, остальные — резерв
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# ВАЖНО: без Accept — Overpass сам ставит формат по [out:json]
HEADERS = {
    "User-Agent": "vegetation-monitor/1.0 (contact@example.com)",
}

TIMEOUT = 30


def _request_json(method: str, url: str, **kwargs) -> Any:
    """GET/POST с таймаутом и обходом SSL-перехвата."""
    kwargs.setdefault("headers", HEADERS)
    kwargs.setdefault("timeout", TIMEOUT)

    try:
        response = requests.request(method, url, **kwargs)
    except requests.exceptions.SSLError:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        kwargs["verify"] = False
        response = requests.request(method, url, **kwargs)

    response.raise_for_status()
    return response.json()


# ----------------------------------------------------------------------
# Поиск региона
# ----------------------------------------------------------------------
def search_regions(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    errors = []

    try:
        return _search_nominatim(query, limit)
    except Exception as exc:
        errors.append(f"Nominatim: {exc}")

    try:
        return _search_photon(query, limit)
    except Exception as exc:
        errors.append(f"Photon: {exc}")

    raise RuntimeError(" | ".join(errors))


def _search_nominatim(query: str, limit: int) -> List[Dict[str, Any]]:
    data = _request_json(
        "GET",
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "limit": limit},
    )

    results = []
    for item in data:
        bbox_raw = item.get("boundingbox")
        bbox = [float(x) for x in bbox_raw] if bbox_raw and len(bbox_raw) == 4 else None

        results.append({
            "name": item.get("display_name"),
            "lat": float(item.get("lat")),
            "lon": float(item.get("lon")),
            "bbox": bbox,
        })

    if not results:
        raise RuntimeError("пустой ответ")

    return results


def _search_photon(query: str, limit: int) -> List[Dict[str, Any]]:
    data = _request_json("GET", PHOTON_URL, params={"q": query, "limit": limit})

    results = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates") or [None, None]

        parts = [props.get(k) for k in ("name", "city", "state", "country") if props.get(k)]
        name = props.get("name") or ", ".join(parts)

        results.append({
            "name": name,
            "lat": float(coords[1]),
            "lon": float(coords[0]),
            "bbox": None,
        })

    if not results:
        raise RuntimeError("пустой ответ")

    return results


# ----------------------------------------------------------------------
# Сельхозконтуры через Overpass
# ----------------------------------------------------------------------


def _overpass_one(url: str, query: str, timeout: int):
    resp = requests.post(
        url,
        data={"data": query},
        headers=HEADERS,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{url}: HTTP {resp.status_code}")
    return resp.json()


def get_farmland_polygons(
    bbox: Tuple[float, float, float, float],
    limit: int = 60,
) -> Dict[str, Any]:
    south, west, north, east = bbox

    query = f"""
    [out:json][timeout:25];
    way
      ["landuse"~"farmland|orchard|vineyard"]
      ({south},{west},{north},{east});
    out geom {limit};
    """

    # ВАЖНО: убираем Accept header — Overpass сам определяет формат через [out:json]
    headers = {
        "User-Agent": "vegetation-monitor/1.0",
    }

    errors = []

    for url in OVERPASS_URLS:
        try:
            # Увеличиваем таймаут до 60 секунд
            response = requests.post(
                url,
                data={"data": query},
                headers=headers,
                timeout=60,
            )

            if response.status_code != 200:
                errors.append(f"{url}: HTTP {response.status_code}")
                continue

            return _overpass_to_geojson(response.json(), limit)

        except requests.exceptions.Timeout:
            errors.append(f"{url}: таймаут (60 с)")
            continue
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue

    raise RuntimeError(" | ".join(errors))

def _overpass_to_geojson(data: Dict[str, Any], limit: int) -> Dict[str, Any]:
    features = []

    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue

        geometry = element.get("geometry")
        if not geometry:
            continue

        coordinates = [[p["lon"], p["lat"]] for p in geometry]

        if len(coordinates) < 3:
            continue

        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        tags = element.get("tags", {})

        features.append({
            "type": "Feature",
            "id": f"way/{element.get('id')}",
            "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            "properties": {
                "source": "openstreetmap",
                "osm_id": element.get("id"),
                "name": tags.get("name", ""),
                "landuse": tags.get("landuse", ""),
                "crop": tags.get("crop", ""),
            },
        })

        if len(features) >= limit:
            break

    return {"type": "FeatureCollection", "features": features}