from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import (
    ID_COL,
    DATE_COL,
    TARGET,
    GAP_FLAG,
    SUBMISSION_PATH,
)
from src.data.loader import load_train, gap_mask
from src.data import osm_client
from src.models.climatology import Climatology
from src.models.interpolation import MLStacker, fill_series
from src.models.anomaly import analyze_series


app = FastAPI(
    title="Vegetation Monitoring API",
    description=(
        "Бэкенд сервиса мониторинга вегетационной динамики "
        "сельскохозяйственных территорий."
    ),
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATE: Dict[str, Any] = {}


class AnalyzeRequest(BaseModel):
    geojson: Dict[str, Any]
    start: str
    end: str


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def _extract_geometry(geojson: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(geojson, dict):
        raise ValueError("GeoJSON должен быть объектом")

    geojson_type = geojson.get("type")

    if geojson_type == "FeatureCollection":
        features = geojson.get("features", [])

        if not features:
            raise ValueError("FeatureCollection пуст")

        geometry = features[0].get("geometry")

        if geometry is None:
            raise ValueError("В FeatureCollection нет геометрии")

        return geometry

    if geojson_type == "Feature":
        geometry = geojson.get("geometry")

        if geometry is None:
            raise ValueError("В Feature нет геометрии")

        return geometry

    return geojson


def _ensure_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if DATE_COL not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Во входных данных нет колонки '{DATE_COL}'",
        )

    out = df.copy()

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL])

    if out.empty:
        return out.reset_index(drop=True)

    if TARGET not in out.columns:
        out[TARGET] = np.nan

    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")

    if ID_COL not in out.columns:
        out[ID_COL] = "UNKNOWN"

    if "year" not in out.columns:
        out["year"] = out[DATE_COL].dt.year

    if "day_of_year" not in out.columns:
        out["day_of_year"] = out[DATE_COL].dt.dayofyear

    out = out.sort_values(DATE_COL).reset_index(drop=True)

    return out


def _payload_from_filled(
    polygon_id: str,
    filled_df: pd.DataFrame,
    climatology: Climatology,
    source: str,
) -> Dict[str, Any]:
    if filled_df is None or filled_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Нет данных для выбранного полигона",
        )

    filled_df = _ensure_date_columns(filled_df)

    if filled_df.empty:
        raise HTTPException(
            status_code=404,
            detail="После обработки данных не осталось пригодных дат",
        )

    if "primary_ndvi_filled" not in filled_df.columns:
        filled_df["primary_ndvi_filled"] = pd.to_numeric(
            filled_df.get(TARGET),
            errors="coerce",
        )

    clim_values = [
        climatology.value(polygon_id, int(doy))
        for doy in filled_df["day_of_year"]
    ]

    clim_mean = np.array([v[0] for v in clim_values], dtype=float)
    clim_std = np.array([v[1] for v in clim_values], dtype=float)

    filled_df["primary_ndvi_filled"] = (
        pd.to_numeric(filled_df["primary_ndvi_filled"], errors="coerce")
        .fillna(pd.Series(clim_mean, index=filled_df.index))
        .clip(0, 1)
    )

    z, status, periods = analyze_series(
        polygon_id=polygon_id,
        g=filled_df,
        clim=climatology,
        value_col="primary_ndvi_filled",
    )

    if TARGET in filled_df.columns:
        raw = pd.to_numeric(filled_df[TARGET], errors="coerce")
    else:
        raw = pd.Series([np.nan] * len(filled_df), index=filled_df.index)

    is_gap = gap_mask(filled_df)

    return {
        "polygon_id": str(polygon_id),
        "source": source,
        "dates": filled_df[DATE_COL].dt.strftime("%Y-%m-%d").tolist(),
        "raw": [
            None if (pd.isna(value) or flag) else round(float(value), 4)
            for value, flag in zip(raw, is_gap)
        ],
        "filled": [
            round(float(value), 4)
            for value in filled_df["primary_ndvi_filled"]
        ],
        "is_gap": [bool(flag) for flag in is_gap],
        "clim_mean": [round(float(value), 4) for value in clim_mean],
        "clim_upper": [
            round(float(mean + 1.96 * std), 4)
            for mean, std in zip(clim_mean, clim_std)
        ],
        "clim_lower": [
            round(float(mean - 1.96 * std), 4)
            for mean, std in zip(clim_mean, clim_std)
        ],
        "z": [round(float(value), 2) for value in z],
        "status": [str(value) for value in status],
        "periods": periods,
    }


def _build_payload(
    polygon_id: str,
    df: pd.DataFrame,
    climatology: Climatology,
    model: MLStacker,
    source: str,
) -> Dict[str, Any]:
    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail="Нет входных данных для анализа",
        )

    df = _ensure_date_columns(df)

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="После обработки входных данных не осталось пригодных дат",
        )

    if TARGET not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Во входных данных нет колонки '{TARGET}'",
        )

    filled_df = fill_series(df, climatology, model)

    return _payload_from_filled(
        polygon_id=polygon_id,
        filled_df=filled_df,
        climatology=climatology,
        source=source,
    )


def _demo_timeseries(
    request: AnalyzeRequest,
    reason: str,
) -> Dict[str, Any]:
    """
    Демо-режим, если нет доступа к Earth Engine
    или для выбранной геометрии не нашлось реальных данных.
    """
    climatology = STATE.get("clim") or Climatology()
    model = STATE.get("stacker")

    try:
        dates = pd.date_range(request.start, request.end, freq="7D")
    except Exception:
        dates = pd.date_range(pd.Timestamp.today(), periods=16, freq="7D")

    if len(dates) < 16:
        start = dates[0] if len(dates) else pd.Timestamp.today()
        dates = pd.date_range(start, periods=16, freq="7D")

    rng = np.random.default_rng(42)

    ndvi_values = []
    temperature_values = []
    precipitation_values = []

    stress_start = max(4, len(dates) // 2)

    for index, date in enumerate(dates):
        mean, std = climatology.value(None, int(date.dayofyear))

        ndvi = mean + rng.normal(0, max(std * 0.35, 0.02))

        temperature = (
            288.0
            + 10.0 * np.sin(2.0 * np.pi * (date.dayofyear - 15) / 365.25)
            + rng.normal(0, 1.5)
        )

        precipitation = max(0.0, rng.gamma(1.0, 2.0))

        # Искусственно создаём демо-аномалию.
        if stress_start <= index < stress_start + 3:
            ndvi -= 0.15
            temperature += 5.0
            precipitation *= 0.2

        ndvi_values.append(float(np.clip(ndvi, 0, 1)))
        temperature_values.append(float(temperature))
        precipitation_values.append(float(precipitation))

    df = pd.DataFrame(
        {
            DATE_COL: dates,
            TARGET: ndvi_values,
            "temperature_2m": temperature_values,
            "tp": precipitation_values,
        }
    )

    hide_mask = rng.random(len(df)) < 0.12
    df.loc[hide_mask, TARGET] = np.nan

    return _build_payload(
        polygon_id="DEMO-POLYGON",
        df=df,
        climatology=climatology,
        model=model,
        source=f"demo (нет реальных данных: {reason[:180]})",
    )


# --------------------------------------------------------------
# Startup
# --------------------------------------------------------------
@app.on_event("startup")
def startup() -> None:
    try:
        train_df = load_train()
    except Exception as exc:
        print(f"WARNING: train_dataset не загружен: {exc}")
        train_df = pd.DataFrame(
            columns=[ID_COL, DATE_COL, TARGET, "year", "day_of_year"]
        )

    try:
        climatology = Climatology.load()
    except Exception:
        climatology = Climatology()

        if not train_df.empty:
            try:
                climatology.fit(train_df)
            except Exception as fit_exc:
                print(f"WARNING: не удалось обучить климатологию: {fit_exc}")

        try:
            climatology.save()
        except Exception as save_exc:
            print(f"WARNING: не удалось сохранить климатологию: {save_exc}")

    try:
        stacker = MLStacker.load()
    except Exception:
        stacker = None

    filled_train = train_df

    if not train_df.empty:
        try:
            filled_train = fill_series(train_df, climatology, stacker)
        except Exception as exc:
            print(f"WARNING: не удалось заполнить обучающие ряды: {exc}")

            filled_train = train_df.copy()
            filled_train["primary_ndvi_filled"] = pd.to_numeric(
                filled_train.get(TARGET),
                errors="coerce",
            )

    STATE.update(
        train=train_df,
        filled_train=filled_train,
        clim=climatology,
        stacker=stacker,
    )


# --------------------------------------------------------------
# Basic endpoints
# --------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Vegetation Monitoring API",
        "endpoints": [
            "GET /api/health",
            "GET /api/polygons",
            "GET /api/timeseries/{polygon_id}",
            "POST /api/analyze",
            "GET /api/regions/search",
            "GET /api/fields",
            "POST /api/inference",
            "GET /api/submission/download",
        ],
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "train_loaded": bool(not STATE.get("train", pd.DataFrame()).empty),
        "model_loaded": STATE.get("stacker") is not None,
    }


# --------------------------------------------------------------
# Scenario 1: dataset polygon
# --------------------------------------------------------------
@app.get("/api/polygons")
def list_polygons() -> List[Dict[str, Any]]:
    train_df = STATE.get("train")

    if train_df is None or train_df.empty:
        return []

    sizes = train_df.groupby(ID_COL).size()

    return [
        {
            "id": str(polygon_id),
            "n_points": int(size),
        }
        for polygon_id, size in sizes.items()
    ]


@app.get("/api/timeseries/{polygon_id}")
def get_timeseries(polygon_id: str):
    filled_train = STATE.get("filled_train")

    if filled_train is None or filled_train.empty:
        raise HTTPException(
            status_code=404,
            detail="Датасет не загружен или пуст",
        )

    polygon_df = filled_train[
        filled_train[ID_COL].astype(str) == str(polygon_id)
    ]

    if polygon_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Полигон {polygon_id} не найден",
        )

    return _payload_from_filled(
        polygon_id=polygon_id,
        filled_df=polygon_df,
        climatology=STATE["clim"],
        source="dataset",
    )


# --------------------------------------------------------------
# Scenario 2: user polygon
# --------------------------------------------------------------
@app.post("/api/analyze")
def analyze_polygon(request: AnalyzeRequest):
    try:
        start = pd.to_datetime(request.start)
        end = pd.to_datetime(request.end)

        if start >= end:
            raise ValueError("start должен быть раньше end")

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Некорректные даты: {exc}",
        )

    try:
        geometry = _extract_geometry(request.geojson)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Некорректный GeoJSON: {exc}",
        )

    try:
        from shapely.geometry import shape

        shapely_geom = shape(geometry)

        if shapely_geom.is_empty or not shapely_geom.is_valid:
            raise ValueError("Пустая или некорректная геометрия")

    except ImportError:
        pass
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Ошибка валидации геометрии: {exc}",
        )

    try:
        from src.data.gee_collector import collect_polygon_data

        satellite_df = collect_polygon_data(
            geojson_geometry=geometry,
            start_date=request.start,
            end_date=request.end,
        )

        if satellite_df.empty:
            raise RuntimeError("Пустой ответ от источника данных")

        satellite_df = satellite_df.copy()
        satellite_df[ID_COL] = "USER-POLYGON"

        return _build_payload(
            polygon_id="USER-POLYGON",
            df=satellite_df,
            climatology=STATE["clim"],
            model=STATE.get("stacker"),
            source="google_earth_engine",
        )

    except Exception as exc:
        return _demo_timeseries(request, reason=str(exc))


# --------------------------------------------------------------
# Scenario 3: region search and farmland polygons
# --------------------------------------------------------------
@app.get("/api/regions/search")
def regions_search(
    q: str = Query(..., description="Название региона"),
    limit: int = Query(5, ge=1, le=20),
):
    try:
        return osm_client.search_regions(q, limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка поиска региона: {exc}",
        )


@app.get("/api/fields")
def fields(
    bbox: str = Query(
        ...,
        description="Границы области: south,west,north,east",
    ),
    limit: int = Query(100, ge=1, le=500),
):
    try:
        south, west, north, east = [
            float(x.strip()) for x in bbox.split(",")
        ]
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="bbox должен быть в формате: south,west,north,east",
        )

    if south >= north or west >= east:
        raise HTTPException(
            status_code=422,
            detail="Некорректный bbox: требуется south < north и west < east",
        )

    try:
        return osm_client.get_farmland_polygons(
            bbox=(south, west, north, east),
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка загрузки полей из OpenStreetMap: {exc}",
        )


# --------------------------------------------------------------
# Scenario 4: technical inference
# --------------------------------------------------------------
@app.post("/api/inference")
def run_inference():
    try:
        from src.inference import main as run_batch_inference

        submission_path = run_batch_inference()

        return {
            "ok": True,
            "submission": submission_path,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка инференса: {exc}",
        )


@app.get("/api/submission/download")
def download_submission():
    path = Path(SUBMISSION_PATH)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="submission.csv ещё не создан. Сначала выполните /api/inference",
        )

    return FileResponse(
        path=path,
        filename="submission.csv",
        media_type="text/csv",
    )