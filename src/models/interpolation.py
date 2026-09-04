from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import PchipInterpolator
from pandas.api.types import CategoricalDtype

from src.config import (
    TARGET,
    ID_COL,
    DATE_COL,
    ARTIFACTS_DIR,
    RANDOM_SEED,
)


LGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

def _to_days(series: pd.Series) -> np.ndarray:
    """Числовая ось времени в днях (для расстояний между датами)."""
    return series.to_numpy(dtype="datetime64[D]").astype(np.float64)


def build_features(df: pd.DataFrame, clim, y: np.ndarray):
    """
    Строит признаки для строк с пропусками.

    Важно:
    - используются только соседние наблюдения того же полигона;
    - климатология;
    - день сезона;
    - тип культуры;
    - статистики полигона.

    Возвращает:
        X, indices
    """
    if df.empty:
        return pd.DataFrame(), []

    y = np.asarray(y, dtype=float)

    if len(y) != len(df):
        raise ValueError("Длина массива y должна совпадать с длиной df")

    X_rows = []
    indices = []

    for polygon_id, g in df.groupby(ID_COL, sort=False):
        g_pos = g.index.to_numpy()
        local_y = y[g_pos]

        t = _to_days(g[DATE_COL])
        doy = g["day_of_year"].to_numpy(dtype=int)

        if "year" in g.columns:
            year = (
                pd.to_numeric(g["year"], errors="coerce")
                .fillna(2000)
                .astype(int)
                .to_numpy()
            )
        else:
            year = np.full(len(g), 2000, dtype=int)

        if "crop_type" in g.columns:
            crop = g["crop_type"].fillna("unk").astype(str).to_numpy()
        else:
            crop = np.full(len(g), "unk", dtype=object)

        clim_values = [clim.value(polygon_id, int(d)) for d in doy]

        clim_mean = np.array([v[0] for v in clim_values], dtype=float)
        clim_std = np.array([v[1] for v in clim_values], dtype=float)

        obs = np.where(~np.isnan(local_y))[0]
        obs_values = local_y[obs]

        if len(obs_values):
            p_mean = float(np.nanmean(obs_values))
            p_std = float(np.nanstd(obs_values))
            p_amp = float(np.nanmax(obs_values) - np.nanmin(obs_values))
        else:
            p_mean = np.nan
            p_std = np.nan
            p_amp = np.nan

        gap_positions = np.where(np.isnan(local_y))[0]

        for k in gap_positions:
            prev = obs[obs < k]
            nxt = obs[obs > k]

            p1 = prev[-1] if len(prev) >= 1 else None
            p2 = prev[-2] if len(prev) >= 2 else None
            n1 = nxt[0] if len(nxt) >= 1 else None
            n2 = nxt[1] if len(nxt) >= 2 else None

            d_prev = float(t[k] - t[p1]) if p1 is not None else np.nan
            d_next = float(t[n1] - t[k]) if n1 is not None else np.nan

            if (
                p1 is not None
                and n1 is not None
                and d_prev > 0
                and d_next > 0
            ):
                w_prev = (1.0 / d_prev) / (1.0 / d_prev + 1.0 / d_next)
                idw = float(
                    w_prev * local_y[p1] + (1.0 - w_prev) * local_y[n1]
                )
            elif p1 is not None:
                w_prev = 1.0
                idw = float(local_y[p1])
            elif n1 is not None:
                w_prev = 0.0
                idw = float(local_y[n1])
            else:
                w_prev = np.nan
                idw = np.nan

            angle = 2.0 * np.pi * float(doy[k]) / 365.25

            X_rows.append(
                {
                    "doy_sin": float(np.sin(angle)),
                    "doy_cos": float(np.cos(angle)),
                    "year": int(year[k]) - 2000,
                    "clim_mean": float(clim_mean[k]) if len(clim_mean) else 0.5,
                    "clim_std": float(clim_std[k]) if len(clim_std) else 0.1,
                    "prev1": float(local_y[p1]) if p1 is not None else np.nan,
                    "prev2": float(local_y[p2]) if p2 is not None else np.nan,
                    "next1": float(local_y[n1]) if n1 is not None else np.nan,
                    "next2": float(local_y[n2]) if n2 is not None else np.nan,
                    "prev1_res": float(local_y[p1] - clim_mean[p1]) if p1 is not None else np.nan,
                    "next1_res": float(local_y[n1] - clim_mean[n1]) if n1 is not None else np.nan,
                    "d_prev": d_prev,
                    "d_next": d_next,
                    "w_prev": float(w_prev),
                    "idw": idw,
                    "p_mean": p_mean,
                    "p_std": p_std,
                    "p_amp": p_amp,
                    "crop_type": str(crop[k]),
                }
            )

            indices.append(int(g_pos[k]))

    if not X_rows:
        return pd.DataFrame(), []

    X = pd.DataFrame(X_rows, index=indices)

    return X, indices


def heuristic_fill(
    polygon_id,
    t: np.ndarray,
    doy: np.ndarray,
    y: np.ndarray,
    clim,
) -> np.ndarray:
    """
    Эвристическое заполнение пропусков.

    Использует:
    - климатологическую норму;
    - интерполяцию сезонных остатков через PCHIP;
    - линейную интерполяцию исходных значений.
    """
    y = np.asarray(y, dtype=float).copy()

    if len(y) == 0:
        return y

    clim_values = [clim.value(polygon_id, int(d)) for d in doy]
    clim_mean = np.array([v[0] for v in clim_values], dtype=float)

    mask = np.isnan(y)

    if not mask.any():
        return np.clip(y, 0, 1)

    obs = ~mask

    if obs.sum() >= 2:
        t_obs = t[obs]
        y_obs = y[obs]
        clim_obs = clim_mean[obs]

        order = np.argsort(t_obs)

        t_obs = t_obs[order]
        y_obs = y_obs[order]
        clim_obs = clim_obs[order]

        unique_t, inverse = np.unique(t_obs, return_inverse=True)

        if len(unique_t) < len(t_obs):
            sums = np.zeros(len(unique_t), dtype=float)
            clim_sums = np.zeros(len(unique_t), dtype=float)
            counts = np.zeros(len(unique_t), dtype=float)

            np.add.at(sums, inverse, y_obs)
            np.add.at(clim_sums, inverse, clim_obs)
            np.add.at(counts, inverse, 1.0)

            t_obs = unique_t
            y_obs = sums / counts
            clim_obs = clim_sums / counts

        if len(t_obs) >= 2:
            residuals = y_obs - clim_obs

            try:
                residual_interp = PchipInterpolator(
                    t_obs,
                    residuals,
                    extrapolate=True,
                )

                predicted_residuals = residual_interp(t[mask])
                linear_values = np.interp(t[mask], t_obs, y_obs)

                y[mask] = (
                    0.6 * (clim_mean[mask] + predicted_residuals)
                    + 0.4 * linear_values
                )
            except Exception:
                y[mask] = np.interp(t[mask], t_obs, y_obs)
        else:
            y[mask] = clim_mean[mask]
    else:
        y[mask] = clim_mean[mask]

    y = np.nan_to_num(y, nan=0.5)
    return np.clip(y, 0, 1)


class MLStacker:
    """
    LightGBM-модель для восстановления пропусков.
    """

    CAT_COL = "crop_type"

    def __init__(self):
        self.model = None
        self.crop_categories_ = ["unk"]

    def _prep(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = X.copy()

        if self.CAT_COL not in X.columns:
            X[self.CAT_COL] = "unk"

        X[self.CAT_COL] = (
            X[self.CAT_COL]
            .fillna("unk")
            .astype(str)
        )

        if fit:
            categories = sorted(set(X[self.CAT_COL]).union({"unk"}))
            self.crop_categories_ = categories
        else:
            categories = self.crop_categories_ or ["unk"]
            X[self.CAT_COL] = X[self.CAT_COL].where(
                X[self.CAT_COL].isin(categories),
                "unk",
            )

        X[self.CAT_COL] = X[self.CAT_COL].astype(
            CategoricalDtype(categories=categories)
        )

        return X

    def fit(self, X, y, Xval=None, yval=None) -> "MLStacker":
        import lightgbm as lgb

        X_prepared = self._prep(X, fit=True)

        self.model = lgb.LGBMRegressor(**LGB_PARAMS)

        if Xval is not None and len(Xval) > 0:
            Xval_prepared = self._prep(Xval, fit=False)

            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(0),
            ]

            self.model.fit(
                X_prepared,
                y,
                eval_set=[(Xval_prepared, yval)],
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_prepared, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не обучена")

        if X is None or len(X) == 0:
            return np.array([], dtype=float)

        X_prepared = self._prep(X, fit=False)
        pred = self.model.predict(X_prepared)

        pred = np.nan_to_num(pred, nan=0.5)
        return np.clip(pred, 0, 1)

    def save(self, path=None) -> None:
        path = Path(path) if path is not None else ARTIFACTS_DIR / "stacker.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "crop_categories_": self.crop_categories_,
        }

        joblib.dump(payload, path)

    @classmethod
    def load(cls, path=None) -> "MLStacker":
        path = Path(path) if path is not None else ARTIFACTS_DIR / "stacker.pkl"

        payload = joblib.load(path)

        obj = cls()

        if isinstance(payload, dict):
            obj.model = payload.get("model")
            obj.crop_categories_ = payload.get("crop_categories_", ["unk"])
        else:
            obj.model = payload
            obj.crop_categories_ = ["unk"]

        return obj


def fill_series(
    df: pd.DataFrame,
    clim,
    stacker: MLStacker = None,
) -> pd.DataFrame:
    """
    Заполняет пропуски в первичном ряде.

    Возвращает копию df с колонкой:
        primary_ndvi_filled
    """
    if df.empty:
        return df.copy()

    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    if TARGET not in df.columns:
        df[TARGET] = np.nan

    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    filled = y.copy()

    if np.isnan(filled).any() and stacker is not None:
        try:
            X, indices = build_features(df, clim, y)

            if len(indices) > 0:
                predictions = stacker.predict(X)
                positions = df.index.get_indexer(indices)
                filled[positions] = predictions
        except Exception as exc:
            print(f"WARNING: ML-заполнение не сработало: {exc}")

    if np.isnan(filled).any():
        for polygon_id, g in df.groupby(ID_COL, sort=False):
            g_pos = g.index.to_numpy()
            local_filled = filled[g_pos]

            need_fill = np.isnan(local_filled)

            if need_fill.any():
                filled_values = heuristic_fill(
                    polygon_id=polygon_id,
                    t=_to_days(g[DATE_COL]),
                    doy=g["day_of_year"].to_numpy(dtype=int),
                    y=y[g_pos],
                    clim=clim,
                )

                filled[g_pos[need_fill]] = filled_values[need_fill]

    out = df.copy()
    out["primary_ndvi_filled"] = np.clip(
        np.nan_to_num(filled, nan=0.5),
        0,
        1,
    )

    return out