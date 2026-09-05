"""
Предвычисление готовых аналитических пакетов по полигонам датасета.

Результат:
    artifacts/polygon_cache/<polygon_id>.json   # готовый payload для фронта
    artifacts/polygon_cache/index.json          # список полигонов с метаданными

Запуск (после обучения):
    python -m src.precompute
"""
import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, ID_COL, DATE_COL, TARGET
from src.data.loader import load_train, gap_mask
from src.models.climatology import Climatology
from src.models.interpolation import fill_series, MLStacker
from src.models.anomaly import analyze_series, _find_col

CACHE_DIR = ARTIFACTS_DIR / "polygon_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _weather_payload(df):
    tcol = _find_col(df, ["temperature", "t2m"])
    pcol = _find_col(df, ["precip", "tp", "rain"])
    if not tcol and not pcol:
        return None
    return {
        "source": "ERA5 (датасет)",
        "temperature": [None if pd.isna(v) else round(float(v), 2) for v in df[tcol]] if tcol else [],
        "precipitation": [None if pd.isna(v) else round(float(v), 2) for v in df[pcol]] if pcol else [],
    }


def build_payload(pid, g, clim) -> dict:
    g = g.sort_values(DATE_COL).reset_index(drop=True)

    z, status, periods = analyze_series(pid, g, clim, "primary_ndvi_filled")

    ms = [clim.value(pid, int(d)) for d in g["day_of_year"]]
    clim_mean = [m for m, _ in ms]
    clim_std = [s for _, s in ms]

    raw = pd.to_numeric(g[TARGET], errors="coerce")
    is_gap = gap_mask(g)

    crop = str(g["crop_type"].iloc[0]) if "crop_type" in g.columns and len(g) else ""

    return {
        "polygon_id": str(pid),
        "source": "dataset (предвычислено)",
        "crop_type": crop,
        "dates": g[DATE_COL].dt.strftime("%Y-%m-%d").tolist(),
        "raw": [None if (pd.isna(v) or f) else round(float(v), 4)
                for v, f in zip(raw, is_gap)],
        "filled": [round(float(v), 4) for v in g["primary_ndvi_filled"]],
        "is_gap": [bool(f) for f in is_gap],
        "clim_mean": [round(float(v), 4) for v in clim_mean],
        "clim_upper": [round(float(m + 1.96 * s), 4) for m, s in zip(clim_mean, clim_std)],
        "clim_lower": [round(float(m - 1.96 * s), 4) for m, s in zip(clim_mean, clim_std)],
        "z": [round(float(v), 2) for v in z],
        "status": [str(s) for s in status],
        "periods": periods,
        "weather": _weather_payload(g),
    }


def main():
    df = load_train()

    try:
        clim = Climatology.load()
    except Exception:
        clim = Climatology().fit(df)
        clim.save()

    try:
        stacker = MLStacker.load()
    except Exception:
        stacker = None

    print("Заполнение рядов…")
    filled = fill_series(df, clim, stacker)

    index = []
    for pid, g in filled.groupby(ID_COL):
        payload = build_payload(pid, g, clim)

        out = CACHE_DIR / f"{pid}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        index.append({
            "id": str(pid),
            "n_points": int(len(g)),
            "crop_type": payload["crop_type"],
            "n_anomalies": len(payload["periods"]),
            "date_min": payload["dates"][0],
            "date_max": payload["dates"][-1],
        })
        print(f"  {pid}: точек={len(g)}, аномалий={len(payload['periods'])}")

    (CACHE_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Готово: {len(index)} полигонов -> {CACHE_DIR}")


if __name__ == "__main__":
    main()