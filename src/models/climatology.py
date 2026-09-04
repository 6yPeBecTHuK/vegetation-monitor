import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    CLIM_WINDOW,
    TARGET,
    ID_COL,
    DATE_COL,
)


class Climatology:
    """
    Климатическая норма по дню сезона.

    Хранит:
    - ожидаемое значение индекса для каждого day-of-year;
    - стандартное отклонение для каждого day-of-year.

    Если для конкретного полигона данных нет, используется глобальная норма.
    """

    def __init__(self, window: int = CLIM_WINDOW):
        self.window = int(window)
        self.stats_: dict = {}
        self.global_: dict = self._default()

    @staticmethod
    def _default() -> dict:
        return {
            "mean": np.full(366, 0.5, dtype=float),
            "std": np.full(366, 0.1, dtype=float),
        }

    def _smooth(self, g: pd.DataFrame) -> dict:
        default = self._default()

        if g is None or g.empty:
            return default

        if TARGET not in g.columns:
            return default

        if "day_of_year" not in g.columns:
            if DATE_COL not in g.columns:
                return default

            g = g.copy()
            g["day_of_year"] = pd.to_datetime(g[DATE_COL]).dt.dayofyear

        g = g.dropna(subset=[TARGET])

        if g.empty:
            return default

        agg = g.groupby("day_of_year")[TARGET].agg(["mean", "std", "count"])

        if agg.empty:
            return default

        mean = np.full(366, np.nan, dtype=float)
        std = np.full(366, np.nan, dtype=float)

        days = agg.index.to_numpy()

        for doy in range(1, 367):
            window_days = days[
                (days >= doy - self.window) &
                (days <= doy + self.window)
            ]

            if len(window_days) == 0:
                continue

            subset = agg.loc[window_days]

            weights = subset["count"].clip(lower=1).astype(float)

            mean[doy - 1] = float(
                np.average(subset["mean"], weights=weights)
            )

            std[doy - 1] = float(
                subset["std"].fillna(0.05).mean()
            )

        mean_series = pd.Series(mean).interpolate(limit_direction="both")
        std_series = pd.Series(std).interpolate(limit_direction="both")

        mean_arr = mean_series.fillna(0.5).to_numpy(dtype=float)
        std_arr = std_series.fillna(0.05).clip(lower=0.02).to_numpy(dtype=float)

        return {
            "mean": mean_arr,
            "std": std_arr,
        }

    def fit(self, df: pd.DataFrame) -> "Climatology":
        if df is None or df.empty:
            return self

        if TARGET not in df.columns:
            return self

        df = df.dropna(subset=[TARGET])

        if df.empty:
            return self

        self.global_ = self._smooth(df)

        if ID_COL in df.columns:
            for polygon_id, g in df.groupby(ID_COL):
                if len(g) >= 5:
                    self.stats_[polygon_id] = self._smooth(g)

        return self

    def norm(self, polygon_id) -> dict:
        return self.stats_.get(polygon_id, self.global_)

    def value(self, polygon_id, day_of_year: int):
        norm = self.norm(polygon_id)

        doy = int(np.clip(int(day_of_year), 1, 366))
        idx = doy - 1

        return float(norm["mean"][idx]), float(norm["std"][idx])

    def save(self, path=None) -> None:
        path = Path(path) if path is not None else ARTIFACTS_DIR / "climatology.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "window": self.window,
            "stats": self.stats_,
            "global": self.global_,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path=None) -> "Climatology":
        path = Path(path) if path is not None else ARTIFACTS_DIR / "climatology.pkl"

        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(window=payload.get("window", CLIM_WINDOW))
        obj.stats_ = payload.get("stats", {})
        obj.global_ = payload.get("global", cls._default())

        return obj