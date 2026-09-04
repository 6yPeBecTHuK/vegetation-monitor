"""
Сбор спутниковых данных через Google Earth Engine.

Основной источник:
    COPERNICUS/S2_SR_HARMONIZED

Дополнительно пытаемся получить метеоданные:
    ECMWF/ERA5_LAND/DAILY

Если Earth Engine недоступен, модуль бросает RuntimeError,
а веб-сервис может переключиться в демо-режим.
"""

from typing import Any, Dict

import pandas as pd


def _init_earth_engine():
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError(
            "Пакет earthengine-api не установлен. "
            "Установите: pip install earthengine-api"
        ) from exc

    try:
        ee.Initialize()
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception as exc:
            raise RuntimeError(
                "Не удалось авторизоваться в Google Earth Engine. "
                "Нужно выполнить ee.Authenticate() и настроить доступ."
            ) from exc

    return ee


def _mask_sentinel2_clouds(image):
    """
    Маскировка облаков для Sentinel-2 по слою QA60.

    Бит 10: opaque clouds
    Бит 11: cirrus
    """
    qa = image.select("QA60")

    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )

    return image.updateMask(mask)


def _add_sentinel2_ndvi(image):
    """
    NDVI для Sentinel-2:

        NDVI = (B8 - B4) / (B8 + B4)

    где:
        B8 - NIR
        B4 - RED
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    return image.addBands(ndvi)


def collect_polygon_data(
    geojson_geometry: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Собирает временной ряд по полигону.

    Возвращает DataFrame:
        date
        primary_ndvi
        temperature_2m   # если удалось получить
        tp               # если удалось получить
    """
    ee = _init_earth_engine()

    geometry = ee.Geometry(geojson_geometry)

    # --------------------------------------------------
    # 1. Sentinel-2 NDVI
    # --------------------------------------------------
    sentinel2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .map(_mask_sentinel2_clouds)
        .map(_add_sentinel2_ndvi)
    )

    def reduce_sentinel2(image):
        values = image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=geometry,
            scale=10,
            maxPixels=1_000_000_000,
        )

        return ee.Feature(None, values).set(
            "date", image.date().format("YYYY-MM-dd")
        )

    sentinel2_info = (
        sentinel2
        .map(reduce_sentinel2)
        .filter(ee.Filter.notNull(["ndvi"]))
        .getInfo()
    ) or {}

    sentinel2_features = sentinel2_info.get("features", [])

    sentinel_rows = []

    for feature in sentinel2_features:
        props = feature.get("properties", {})

        if props.get("ndvi") is None:
            continue

        if props.get("date") is None:
            continue

        sentinel_rows.append(
            {
                "date": props.get("date"),
                "primary_ndvi": float(props.get("ndvi")),
            }
        )

    if not sentinel_rows:
        raise RuntimeError(
            "Не удалось получить пригодные наблюдения Sentinel-2 "
            "для выбранного полигона и периода."
        )

    df = pd.DataFrame(sentinel_rows)
    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------
    # 2. ERA5-Land weather, если получится
    # --------------------------------------------------
    try:
        era5 = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select(["temperature_2m", "tp"])
        )

        def reduce_era5(image):
            values = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10000,
                maxPixels=1_000_000_000,
            )

            return ee.Feature(None, values).set(
                "date", image.date().format("YYYY-MM-dd")
            )

        era5_info = era5.map(reduce_era5).getInfo() or {}
        era5_features = era5_info.get("features", [])

        era5_rows = []

        for feature in era5_features:
            props = feature.get("properties", {})

            if props.get("date") is None:
                continue

            era5_rows.append(
                {
                    "date": props.get("date"),
                    "temperature_2m": props.get("temperature_2m"),
                    "tp": props.get("tp"),
                }
            )

        if era5_rows:
            df_era5 = pd.DataFrame(era5_rows)
            df_era5["date"] = pd.to_datetime(df_era5["date"])

            df = df.merge(df_era5, on="date", how="left")

    except Exception:
        # Метеоданные не критичны для базового сценария.
        # Если ERA5 не получилась, продолжаем без неё.
        pass

    df = df.sort_values("date").reset_index(drop=True)

    return df