import numpy as np
import pandas as pd

from src.config import (
    Z_NORMAL,
    Z_STRESS,
    MIN_ANOMALY_LEN,
    MAX_PERIOD_GAP_DAYS,
    TARGET,
    DATE_COL,
)


STATUS_LABELS = {
    "normal": "штатное развитие",
    "stress": "угнетение биомассы",
    "critical": "критическая аномалия",
}


def _find_col(df: pd.DataFrame, keys):
    for col in df.columns:
        col_lower = str(col).lower()

        if any(key in col_lower for key in keys):
            return col

    return None


def _interpret(sub: pd.DataFrame, g_all: pd.DataFrame) -> str:
    """
    Пытается объяснить аномалию через температуру и осадки.
    """
    messages = []

    temp_col = _find_col(g_all, ["temperature", "t2m"])
    precip_col = _find_col(g_all, ["precip", "tp", "rain"])

    if temp_col:
        sub_temp = pd.to_numeric(sub[temp_col], errors="coerce").dropna()
        all_temp = pd.to_numeric(g_all[temp_col], errors="coerce").dropna()

        if len(sub_temp) > 0 and len(all_temp) > 5:
            temp_std = float(all_temp.std())

            if np.isfinite(temp_std) and temp_std > 1e-6:
                z_temp = (
                    float(sub_temp.mean()) - float(all_temp.mean())
                ) / temp_std

                if z_temp > 1.0:
                    messages.append(
                        f"температурный стресс (+{z_temp:.1f}σ к норме)"
                    )
                elif z_temp < -1.0:
                    messages.append(
                        f"холодовой стресс ({z_temp:.1f}σ)"
                    )

    if precip_col:
        sub_precip = pd.to_numeric(sub[precip_col], errors="coerce").dropna()
        all_precip = pd.to_numeric(g_all[precip_col], errors="coerce").dropna()

        if len(sub_precip) > 0 and len(all_precip) > 0:
            all_mean = float(all_precip.mean())

            if np.isfinite(all_mean) and abs(all_mean) > 1e-9:
                ratio = float(sub_precip.mean()) / max(all_mean, 1e-9)

                if ratio < 0.5:
                    messages.append("дефицит осадков — вероятна засуха")
                elif ratio > 2.0:
                    messages.append("избыточное увлажнение")

    if not messages:
        messages.append(
            "устойчивое отрицательное отклонение от климатической нормы; "
            "проверьте погодный контекст, фазу развития культуры и качество данных"
        )

    return "; ".join(messages)


def _make_period(
    sub: pd.DataFrame,
    z: np.ndarray,
    status: np.ndarray,
    g_all: pd.DataFrame,
    value_col: str,
) -> dict:
    severity = "critical" if np.any(np.asarray(status) == "critical") else "stress"

    return {
        "start": str(sub[DATE_COL].iloc[0].date()),
        "end": str(sub[DATE_COL].iloc[-1].date()),
        "days": int((sub[DATE_COL].iloc[-1] - sub[DATE_COL].iloc[0]).days + 1),
        "severity": severity,
        "label": STATUS_LABELS[severity],
        "z_min": round(float(np.min(z)), 2),
        "z_mean": round(float(np.mean(z)), 2),
        "ndvi_min": round(float(sub[value_col].min()), 3),
        "interpretation": _interpret(sub, g_all),
    }


def analyze_series(
    polygon_id,
    g: pd.DataFrame,
    clim,
    value_col: str = "primary_ndvi_filled",
):
    """
    Возвращает:
        z,
        status,
        periods
    """
    if g.empty:
        return np.array([]), np.array([]), []

    g = g.sort_values(DATE_COL).reset_index(drop=True)

    if value_col not in g.columns:
        if TARGET in g.columns:
            value_col = TARGET
        else:
            raise ValueError("Нет колонки с заполненным значением индекса")

    y = pd.to_numeric(g[value_col], errors="coerce").to_numpy(dtype=float)

    clim_values = [clim.value(polygon_id, int(d)) for d in g["day_of_year"]]

    clim_mean = np.array([v[0] for v in clim_values], dtype=float)
    clim_std = np.array([v[1] for v in clim_values], dtype=float)

    y = np.where(np.isnan(y), clim_mean, y)

    z = (y - clim_mean) / np.clip(clim_std, 1e-3, None)

    status = np.full(len(g), "normal", dtype=object)
    status[(z < Z_NORMAL) & (z >= Z_STRESS)] = "stress"
    status[z < Z_STRESS] = "critical"

    periods = []

    i = 0
    n = len(g)

    while i < n:
        if status[i] == "normal":
            i += 1
            continue

        j = i

        while j + 1 < n and status[j + 1] != "normal":
            gap_days = (g[DATE_COL].iloc[j + 1] - g[DATE_COL].iloc[j]).days

            if gap_days > MAX_PERIOD_GAP_DAYS:
                break

            j += 1

        if j - i + 1 >= MIN_ANOMALY_LEN:
            sub = g.iloc[i:j + 1].copy()
            sub[value_col] = y[i:j + 1]

            periods.append(
                _make_period(
                    sub=sub,
                    z=z[i:j + 1],
                    status=status[i:j + 1],
                    g_all=g,
                    value_col=value_col,
                )
            )

        i = j + 1

    return z, status, periods