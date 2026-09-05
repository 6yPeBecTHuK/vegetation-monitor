"""
Спутниковые наблюдения по произвольной фигуре.

Используем NASA ORNL DAAC MODIS Subsets API.
Авторизация не нужна.

Важно:
ORNL ограничивает один запрос максимум 10 MODIS-композитами,
поэтому длинный период режем на чанки.
"""

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import requests

ORNL_BASE = "https://modis.ornl.gov/rst/api/v1"

HEADERS = {
    "User-Agent": "vegetation-monitor/1.0",
    "Accept": "application/json",
}


def _to_ornl_date(date_value) -> str:
    """
    ORNL ждёт дату в формате AYYYYDDD.
    Например:
        2024-04-01 -> A2024092
    """
    ts = pd.to_datetime(date_value)
    return f"A{ts.year}{ts.dayofyear:03d}"


def _polygon_centroid_and_radius(
    geojson_geometry: Dict[str, Any],
) -> Tuple[float, float, int]:
    """
    Возвращает:
        lat, lon, km

    ORNL берёт квадратное окно вокруг точки:
        kmAboveBelow / kmLeftRight
    """
    from shapely.geometry import shape

    geom = shape(geojson_geometry)

    if geom.is_empty or not geom.is_valid:
        raise ValueError("Пустая или некорректная геометрия")

    point = geom.representative_point()

    minx, miny, maxx, maxy = geom.bounds

    km_w = abs(maxx - minx) * 111.32 * math.cos(math.radians(point.y))
    km_h = abs(maxy - miny) * 110.57

    half = max(km_w, km_h) / 2.0

    # ORNL subset не надо делать слишком большим
    km = int(np.clip(math.ceil(half), 0, 5))

    return float(point.y), float(point.x), km


def _mean_valid_modis_values(values) -> float:
    """
    MODIS NDVI хранится как int со scale_factor=0.0001.
    Fill values обычно <= -3000.
    """
    if values is None:
        return np.nan

    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return np.nan

    arr = arr[np.isfinite(arr)]
    arr = arr[arr > -3000]

    if arr.size == 0:
        return np.nan

    ndvi = arr * 0.0001
    ndvi = ndvi[(ndvi >= -1.0) & (ndvi <= 1.0)]

    if ndvi.size == 0:
        return np.nan

    return float(np.nanmean(ndvi))


def _ornl_fetch_band_single(
    product: str,
    band: str,
    lat: float,
    lon: float,
    km: int,
    start,
    end,
) -> Dict[str, float]:
    """
    Один короткий запрос в ORNL.
    Не должен превышать лимит ~10 композитов.
    """
    params = {
        "band": band,
        "latitude": round(float(lat), 5),
        "longitude": round(float(lon), 5),
        "startDate": _to_ornl_date(start),
        "endDate": _to_ornl_date(end),
        "kmAboveBelow": int(km),
        "kmLeftRight": int(km),
    }

    url = f"{ORNL_BASE}/{product}/subset"

    resp = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"{resp.status_code} {resp.reason}: {resp.text[:300]}"
        )

    payload = resp.json() or {}
    subset = payload.get("subset") or []

    if not subset:
        return {}

    per_date = {}

    for item in subset:
        calendar_date = item.get("calendar_date")
        data_values = item.get("data")

        if not calendar_date:
            continue

        value = _mean_valid_modis_values(data_values)

        if np.isfinite(value):
            per_date[calendar_date] = value

    return per_date


def _ornl_fetch_band(
    product: str,
    band: str,
    lat: float,
    lon: float,
    km: int,
    start,
    end,
) -> Dict[str, float]:
    """
    Длинный период режем на куски.

    Почему:
    ORNL MODIS Subsets API возвращает ошибку:
        exceeds maximum subset tiles support of 10

    Берём чанки по 120 дней — это безопасно меньше 10 композитов MOD13Q1.
    """
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    all_values: Dict[str, float] = {}

    chunk_start = s
    chunk_days = 120

    errors = []

    while chunk_start <= e:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), e)

        try:
            part = _ornl_fetch_band_single(
                product=product,
                band=band,
                lat=lat,
                lon=lon,
                km=km,
                start=chunk_start,
                end=chunk_end,
            )
            all_values.update(part)

        except Exception as exc:
            errors.append(
                f"{chunk_start:%Y-%m-%d}..{chunk_end:%Y-%m-%d}: {exc}"
            )

        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not all_values:
        raise RuntimeError("нет пригодных значений NDVI; " + " | ".join(errors))

    return all_values


def _ornl_modis_series(
    geojson_geometry: Dict[str, Any],
    start: str,
    end: str,
):
    """
    Получает MODIS NDVI через ORNL.

    Используем MOD13Q1:
        Terra MODIS, 250 м, 16-дневные NDVI-композиты.
    """
    lat, lon, km = _polygon_centroid_and_radius(geojson_geometry)

    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    # MODIS доступен с 2000 года и с задержкой.
    max_end = pd.Timestamp.today() - pd.Timedelta(days=45)
    min_start = pd.Timestamp("2000-03-01")

    clamped_note = ""

    if e > max_end:
        e = max_end
        clamped_note = (
            f"; период автоматически ограничен по {e:%Y-%m-%d} "
            "(запаздывание MODIS ~1.5 мес)"
        )

    if s < min_start:
        s = min_start

    if s >= e:
        raise RuntimeError(
            "период слишком свежий для MODIS — "
            "выберите более ранний период"
        )

    product = "MOD13Q1"
    band = "250m_16_days_NDVI"

    series = _ornl_fetch_band(
        product=product,
        band=band,
        lat=lat,
        lon=lon,
        km=km,
        start=s,
        end=e,
    )

    rows = []

    for date_str, ndvi in sorted(series.items()):
        rows.append(
            {
                "date": pd.to_datetime(date_str),
                "primary_ndvi": float(ndvi),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("ORNL вернул пустой DataFrame")

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    meta = {
        "source": "NASA ORNL DAAC MODIS Subsets API",
        "sensor": "MODIS Terra MOD13Q1, NDVI, 250 м, 16-дневные композиты",
        "note": (
            f"центр сектора: {lat:.4f}, {lon:.4f}; "
            f"окно ORNL: ±{km} км; "
            f"период: {s:%Y-%m-%d}..{e:%Y-%m-%d}; "
            "длинные периоды автоматически разбиваются на несколько запросов"
            + clamped_note
        ),
    }

    return df, meta


def search_satellite_series(
    geojson_geometry: Dict[str, Any],
    start: str,
    end: str,
):
    """
    Точка входа для webapp.

    Возвращает:
        df:
            date
            primary_ndvi

        meta:
            source
            sensor
            note
    """
    return _ornl_modis_series(
        geojson_geometry=geojson_geometry,
        start=start,
        end=end,
    )