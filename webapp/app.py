"""Vegetation Monitoring API — стабильная версия."""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    ID_COL,
    DATE_COL,
    TARGET,
    GAP_FLAG,
    SUBMISSION_PATH,
    ARTIFACTS_DIR,
)
from src.data.loader import load_train, gap_mask
from src.data import osm_client
from src.models.climatology import Climatology
from src.models.interpolation import MLStacker, fill_series
from src.models.anomaly import analyze_series, _find_col

# защитные импорты: сервис стартует, даже если модулей нет
try:
    from src.data.weather_client import get_weather_for_polygon, polygon_centroid
except Exception:
    get_weather_for_polygon = None
    polygon_centroid = None

try:
    from src.data.satellite_provider import search_satellite_series
except Exception:
    def search_satellite_series(geometry, start, end):
        raise RuntimeError("satellite_provider не подключён")

app = FastAPI(title="Vegetation Monitoring API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATE: Dict[str, Any] = {}
_FIELDS_CACHE: Dict[Any, Any] = {}


class AnalyzeRequest(BaseModel):
    geojson: Dict[str, Any]
    start: str
    end: str

POLYGON_CACHE_DIR = ARTIFACTS_DIR / "polygon_cache"
_POLY_MEMORY: Dict[str, Any] = {}
# ------------------------------------------------------------------
def _extract_geometry(geojson: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(geojson, dict):
        raise ValueError("GeoJSON должен быть объектом")
    t = geojson.get("type")
    if t == "FeatureCollection":
        feats = geojson.get("features", [])
        if not feats:
            raise ValueError("FeatureCollection пуст")
        return feats[0].get("geometry") or {}
    if t == "Feature":
        return geojson.get("geometry") or {}
    return geojson


def _ensure_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if DATE_COL not in df.columns:
        raise HTTPException(422, f"Нет колонки '{DATE_COL}'")
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", format="%Y-%m-%d")
    out = out.dropna(subset=[DATE_COL])
    if out.empty:
        return out.reset_index(drop=True)
    if TARGET not in out.columns:
        out[TARGET] = np.nan
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    if ID_COL not in out.columns:
        out[ID_COL] = "UNKNOWN"
    out["year"] = out[DATE_COL].dt.year
    out["day_of_year"] = out[DATE_COL].dt.dayofyear
    return out.sort_values(DATE_COL).reset_index(drop=True)


def _weather_payload(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    tcol = _find_col(df, ["temperature", "t2m"])
    pcol = _find_col(df, ["precip", "tp", "rain"])
    if not tcol and not pcol:
        return None
    source = (
        "ERA5 (датасет)"
        if (tcol and str(tcol).lower().startswith("era5"))
        else "Open-Meteo (ERA5), центр сектора"
    )
    return {
        "source": source,
        "temperature": [None if pd.isna(v) else round(float(v), 2) for v in df[tcol]] if tcol else [],
        "precipitation": [None if pd.isna(v) else round(float(v), 2) for v in df[pcol]] if pcol else [],
    }


def _payload_from_filled(polygon_id, filled_df, climatology, source):
    if filled_df is None or filled_df.empty:
        raise HTTPException(404, "Нет данных для полигона")
    filled_df = _ensure_date_columns(filled_df)
    if filled_df.empty:
        raise HTTPException(404, "Нет пригодных дат")
    if "primary_ndvi_filled" not in filled_df.columns:
        filled_df["primary_ndvi_filled"] = pd.to_numeric(filled_df.get(TARGET), errors="coerce")

    clim_values = [climatology.value(polygon_id, int(d)) for d in filled_df["day_of_year"]]
    clim_mean = np.array([v[0] for v in clim_values], dtype=float)
    clim_std = np.array([v[1] for v in clim_values], dtype=float)

    filled_df["primary_ndvi_filled"] = (
        pd.to_numeric(filled_df["primary_ndvi_filled"], errors="coerce")
        .fillna(pd.Series(clim_mean, index=filled_df.index))
        .clip(0, 1)
    )

    z, status, periods = analyze_series(polygon_id, filled_df, climatology, "primary_ndvi_filled")

    raw = pd.to_numeric(filled_df[TARGET], errors="coerce") if TARGET in filled_df else pd.Series([np.nan] * len(filled_df))
    is_gap = gap_mask(filled_df)

    return {
        "polygon_id": str(polygon_id),
        "source": source,
        "dates": filled_df[DATE_COL].dt.strftime("%Y-%m-%d").tolist(),
        "raw": [None if (pd.isna(v) or f) else round(float(v), 4) for v, f in zip(raw, is_gap)],
        "filled": [round(float(v), 4) for v in filled_df["primary_ndvi_filled"]],
        "is_gap": [bool(f) for f in is_gap],
        "clim_mean": [round(float(v), 4) for v in clim_mean],
        "clim_upper": [round(float(m + 1.96 * s), 4) for m, s in zip(clim_mean, clim_std)],
        "clim_lower": [round(float(m - 1.96 * s), 4) for m, s in zip(clim_mean, clim_std)],
        "z": [round(float(v), 2) for v in z],
        "status": [str(s) for s in status],
        "periods": periods,
        "weather": _weather_payload(filled_df),
    }


def _build_payload(polygon_id, df, climatology, model, source):
    df = _ensure_date_columns(df)
    if df.empty:
        raise HTTPException(422, "Нет пригодных дат")
    filled_df = fill_series(df, climatology, model)
    return _payload_from_filled(polygon_id, filled_df, climatology, source)


def _demo_timeseries(request, reason, weather_df=None):
    climatology = STATE.get("clim") or Climatology()
    model = STATE.get("stacker")

    has_weather = weather_df is not None and not weather_df.empty
    if has_weather:
        dates = pd.to_datetime(weather_df["date"])
    else:
        try:
            dates = pd.date_range(request.start, request.end, freq="7D")
        except Exception:
            dates = pd.date_range(pd.Timestamp.today(), periods=16, freq="7D")
        if len(dates) < 16:
            dates = pd.date_range(dates[0] if len(dates) else pd.Timestamp.today(), periods=16, freq="7D")

    rng = np.random.default_rng(42)
    ndvi_values = []
    stress_start = max(4, len(dates) // 2)
    for index, date in enumerate(dates):
        mean, std = climatology.value(None, int(date.dayofyear))
        ndvi = mean + rng.normal(0, max(std * 0.35, 0.02))
        if stress_start <= index < stress_start + 3:
            ndvi -= 0.15
        ndvi_values.append(float(np.clip(ndvi, 0, 1)))

    df = pd.DataFrame({DATE_COL: dates, TARGET: ndvi_values})

    if has_weather:
        df = df.merge(weather_df, on="date", how="left")
    else:
        temps, precs = [], []
        for index, date in enumerate(dates):
            t = 288.0 + 10.0 * np.sin(2 * np.pi * (date.dayofyear - 15) / 365.25) + rng.normal(0, 1.5)
            p = max(0.0, rng.gamma(1.0, 2.0))
            if stress_start <= index < stress_start + 3:
                t += 5.0
                p *= 0.2
            temps.append(float(t))
            precs.append(float(p))
        df["temperature_2m"] = temps
        df["precipitation"] = precs

    hide = rng.random(len(df)) < 0.12
    df.loc[hide, TARGET] = np.nan

    return _build_payload("DEMO-POLYGON", df, climatology, model,
                          f"demo (нет спутниковых данных: {reason[:180]})")


# ------------------------------------------------------------------
@app.on_event("startup")
def startup() -> None:
    try:
        train_df = load_train()
    except Exception as exc:
        print(f"WARNING: train не загружен: {exc}")
        train_df = pd.DataFrame(columns=[ID_COL, DATE_COL, TARGET, "year", "day_of_year"])

    try:
        climatology = Climatology.load()
    except Exception:
        climatology = Climatology()
        if not train_df.empty:
            climatology.fit(train_df)
        try:
            climatology.save()
        except Exception:
            pass

    try:
        stacker = MLStacker.load()
    except Exception:
        stacker = None

    filled_train = train_df
    if not train_df.empty:
        try:
            filled_train = fill_series(train_df, climatology, stacker)
        except Exception as exc:
            print(f"WARNING: {exc}")
            filled_train = train_df.copy()
            filled_train["primary_ndvi_filled"] = pd.to_numeric(filled_train.get(TARGET), errors="coerce")

    STATE.update(train=train_df, filled_train=filled_train, clim=climatology, stacker=stacker)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "train_loaded": bool(not STATE.get("train", pd.DataFrame()).empty),
        "model_loaded": STATE.get("stacker") is not None,
    }


@app.get("/api/polygons")
def list_polygons():
    # 1) готовые данные с диска — мгновенно
    idx_path = POLYGON_CACHE_DIR / "index.json"
    if idx_path.exists():
        try:
            return json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2) фолбэк: живой подсчёт из датасета
    train_df = STATE.get("train")
    if train_df is None or train_df.empty:
        return []
    sizes = train_df.groupby(ID_COL).size()
    return [{"id": str(pid), "n_points": int(n)} for pid, n in sizes.items()]


@app.get("/api/timeseries/{polygon_id}")
def get_timeseries(polygon_id: str):
    # 1) память
    if polygon_id in _POLY_MEMORY:
        return _POLY_MEMORY[polygon_id]

    # 2) диск
    path = POLYGON_CACHE_DIR / f"{polygon_id}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _POLY_MEMORY[polygon_id] = data
            return data
        except Exception:
            pass

    # 3) фолбэк: живой пересчёт
    filled_train = STATE.get("filled_train")
    if filled_train is None or filled_train.empty:
        raise HTTPException(404, "Датасет не загружен")
    polygon_df = filled_train[filled_train[ID_COL].astype(str) == str(polygon_id)]
    if polygon_df.empty:
        raise HTTPException(404, f"Полигон {polygon_id} не найден")
    return _payload_from_filled(polygon_id, polygon_df, STATE["clim"], "dataset")

@app.post("/api/analyze")
def analyze_polygon(request: AnalyzeRequest):
    try:
        start = pd.to_datetime(request.start)
        end = pd.to_datetime(request.end)
        if start >= end:
            raise ValueError()
    except Exception:
        raise HTTPException(422, "Некорректные даты")

    try:
        geometry = _extract_geometry(request.geojson)
    except Exception as exc:
        raise HTTPException(422, f"Некорректный GeoJSON: {exc}")

    weather_df = None
    centroid = None
    if polygon_centroid is not None and get_weather_for_polygon is not None:
        try:
            centroid = polygon_centroid(geometry)
            weather_df = get_weather_for_polygon(geometry, request.start, request.end)
        except Exception as exc:
            print(f"WARNING: погода не получена: {exc}")

    try:
        satellite_df, meta = search_satellite_series(geometry, request.start, request.end)
        satellite_df = satellite_df.copy()
        satellite_df[ID_COL] = "USER-POLYGON"
        if weather_df is not None and not weather_df.empty:
            for col in ("temperature_2m", "precipitation"):
                if col not in satellite_df.columns:
                    satellite_df = satellite_df.merge(weather_df[["date", col]], on="date", how="left")
        payload = _build_payload("USER-POLYGON", satellite_df, STATE["clim"],
                                 STATE.get("stacker"), meta["source"])
        payload["satellite"] = {**meta, "n_obs": int(len(satellite_df))}
    except Exception as exc:
        payload = _demo_timeseries(request, str(exc), weather_df)
        payload["satellite"] = None

    if centroid is not None:
        payload["centroid"] = [round(centroid[0], 4), round(centroid[1], 4)]

    return payload


@app.get("/api/regions/search")
def regions_search(q: str = Query(...), limit: int = Query(5, ge=1, le=20)):
    try:
        return osm_client.search_regions(q, limit=limit)
    except Exception as exc:
        raise HTTPException(502, f"Ошибка поиска региона: {exc}")


OSM_CACHE_DIR = ARTIFACTS_DIR / "osm_cache"
OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/fields")
def fields(
    bbox: str = Query(..., description="south,west,north,east"),
    limit: int = Query(60, ge=1, le=200),
):
    try:
        south, west, north, east = [float(x.strip()) for x in bbox.split(",")]
    except Exception:
        raise HTTPException(422, "bbox: south,west,north,east")

    if south >= north or west >= east:
        raise HTTPException(422, "Некорректный bbox")

    # ------------------------------------------------------------
    # Ограничиваем площадь окна: для больших регионов (весь край)
    # берём квадрат 2°×2° вокруг центра видимой области.
    # Overpass на полном крае часто падает или возвращается пустым.
    # ------------------------------------------------------------
    MAX_SPAN = 2.0
    if (north - south) > MAX_SPAN or (east - west) > MAX_SPAN:
        c_lat = (north + south) / 2.0
        c_lon = (east + west) / 2.0
        south, north = c_lat - MAX_SPAN / 2.0, c_lat + MAX_SPAN / 2.0
        west, east = c_lon - MAX_SPAN / 2.0, c_lon + MAX_SPAN / 2.0

    key = f"{south:.2f}_{west:.2f}_{north:.2f}_{east:.2f}_{limit}"

    # 1) in-memory кэш
    if key in _FIELDS_CACHE:
        return _FIELDS_CACHE[key]

    # 2) disk-кэш — повторная загрузка того же места мгновенна
    cache_path = OSM_CACHE_DIR / (key + ".geojson")
    if cache_path.exists():
        try:
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            _FIELDS_CACHE[key] = result
            return result
        except Exception:
            pass

    # 3) запрос в OSM
    try:
        result = osm_client.get_farmland_polygons(
            (south, west, north, east), limit
        )
    except Exception as exc:
        raise HTTPException(502, f"Ошибка OSM: {exc}")

    try:
        cache_path.write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass

    _FIELDS_CACHE[key] = result
    return result

@app.post("/api/inference")
def run_inference():
    try:
        from src.inference import main as run_batch
        return {"ok": True, "submission": run_batch()}
    except Exception as exc:
        raise HTTPException(500, f"Ошибка инференса: {exc}")


@app.get("/api/submission/download")
def download_submission():
    path = Path(SUBMISSION_PATH)
    if not path.exists():
        raise HTTPException(404, "submission.csv ещё не создан")
    return FileResponse(path, filename="submission.csv", media_type="text/csv")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")