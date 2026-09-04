import numpy as np
import pandas as pd

from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    TARGET,
    GAP_FLAG,
    ID_COL,
    DATE_COL,
)

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "да"}

# Фиксированный порядок 20 колонок файлов организаторов
EXPECTED_COLS = [
    ID_COL, DATE_COL,
    "s2_ndvi", "s2_evi", "s2_ndwi",
    "landsat_ndvi", "landsat_evi", "landsat_ndwi",
    "modis_ndvi", "modis_evi",
    "era5_temperature_2m", "era5_tp",
    "year", TARGET, "day_of_year",
    "climatology_mean", "climatology_std",
    "p_reference_years", GAP_FLAG, "crop_type",
]


def boolean_mask(series) -> np.ndarray:
    """Аккуратно приводит колонку к булеву массиву (True/False, 1/0, строки)."""
    if series is None:
        return np.zeros(0, dtype=bool)

    s = pd.Series(series).copy()

    if s.empty:
        return np.zeros(len(s), dtype=bool)

    if s.dtype == bool:
        return s.fillna(False).to_numpy(dtype=bool)

    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0).to_numpy(dtype=bool)

    normalized = s.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(_TRUE_VALUES).to_numpy(dtype=bool)


def _read_dataset(path) -> pd.DataFrame:
    """
    Читает CSV устойчиво к состоянию шапки:
    - первая строка не похожа на данные (не начинается с AOI) -> считаем её
      шапкой и пропускаем, имена назначаем по позициям;
    - первая строка похожа на данные -> шапки нет, имена по позициям;
    - число колонок нестандартное (например, train с ndvi_zscore/status) ->
      читаем как есть с собственной шапкой файла.
    """
    try:
        sniff = pd.read_csv(
            path, nrows=5, header=None, dtype=str, encoding="utf-8-sig"
        )
    except Exception:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

    if sniff.empty:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

    ncols = sniff.shape[1]
    first_val = str(sniff.iloc[0, 0]).strip().lower()
    first_row_is_data = first_val.startswith("aoi")

    if ncols == len(EXPECTED_COLS):
        if first_row_is_data:
            return pd.read_csv(
                path, header=None, names=EXPECTED_COLS,
                low_memory=False, encoding="utf-8-sig",
            )
        return pd.read_csv(
            path, header=None, names=EXPECTED_COLS, skiprows=1,
            low_memory=False, encoding="utf-8-sig",
        )

    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if DATE_COL not in df.columns:
        raise KeyError(f"В данных нет обязательной колонки '{DATE_COL}'")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL], errors="coerce", format="%Y-%m-%d"
    )
    df = df.dropna(subset=[DATE_COL])

    if TARGET not in df.columns:
        df[TARGET] = np.nan

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if ID_COL not in df.columns:
        df[ID_COL] = "UNKNOWN"

    # year и day_of_year ВСЕГДА пересчитываем из даты:
    # в тесте они могут быть замаскированы или прочитаны как строки
    df["year"] = df[DATE_COL].dt.year
    df["day_of_year"] = df[DATE_COL].dt.dayofyear

    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    return df


def load_train(path=None) -> pd.DataFrame:
    df = _read_dataset(path or TRAIN_CSV)
    return _prepare(df)


def load_test(path=None) -> pd.DataFrame:
    df = _read_dataset(path or TEST_CSV)
    df = _prepare(df)

    if GAP_FLAG in df.columns:
        mask = boolean_mask(df[GAP_FLAG])
        df.loc[mask, TARGET] = np.nan

    return df


def gap_mask(df: pd.DataFrame) -> np.ndarray:
    """Булев массив: какие строки являются контрольными пропусками."""
    if df is None or df.empty:
        return np.zeros(0, dtype=bool)

    if GAP_FLAG not in df.columns:
        return np.zeros(len(df), dtype=bool)

    return boolean_mask(df[GAP_FLAG])