"""
Привязка погоды к выбранному сектору (полигону) через открытый API Open-Meteo.

Ключ API не нужен. Исторический архив Open-Meteo построен на реанализе ERA5,
что соответствует рекомендациям постановки (ERA5/ERA5-Land).

Источники:
    архив:    https://archive-api.open-meteo.com/v1/archive
    текущая:  https://api.open-meteo.com/v1/forecast
"""

from typing import Any, Dict, Tuple

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = "temperature_2m_mean,precipitation_sum"

_CACHE: Dict[Tuple, pd.DataFrame] = {}


def polygon_centroid(geojson_geometry: Dict[str, Any]) -> Tuple[float, float]:
    """Возвращает (lat, lon) репрезентативной точки полигона."""
    from shapely.geometry import shape

    point = shape(geojson_geometry).representative_point()
    return float(point.y), float(point.x)


def _fetch_daily(url: str, lat: float, lon: float,
                 start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": DAILY_VARS,
        "timezone": "UTC",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    daily = (response.json() or {}).get("daily") or {}
    times = daily.get("time") or []

    if not times:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "date": pd.to_datetime(times),
            "temperature_2m": pd.to_numeric(
                daily.get("temperature_2m_mean"), errors="coerce"
            ),
            "precipitation": pd.to_numeric(
                daily.get("precipitation_sum"), errors="coerce"
            ),
        }
    )


def get_weather_for_polygon(
    geojson_geometry: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Суточный ряд погоды по центру полигона:
        date, temperature_2m (°C), precipitation (мм)

    Сначала пробуем исторический архив, затем — forecast-эндпоинт
    (он отдаёт последние ~3 месяца и прогноз).
    """
    lat, lon = polygon_centroid(geojson_geometry)

    key = (round(lat, 2), round(lon, 2), str(start_date), str(end_date))
    if key in _CACHE:
        return _CACHE[key]

    errors = []
    df = pd.DataFrame()

    for url in (ARCHIVE_URL, FORECAST_URL):
        try:
            df = _fetch_daily(url, lat, lon, start_date, end_date)
            if not df.empty:
                break
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if df.empty:
        raise RuntimeError("; ".join(errors) or "Open-Meteo: пустой ответ")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    _CACHE[key] = df
    return df