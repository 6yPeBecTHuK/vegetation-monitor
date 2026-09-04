from pathlib import Path

import pandas as pd

from src.config import (
    SUBMISSION_PATH,
    ID_COL,
    DATE_COL,
    TARGET,
    GAP_FLAG,
)
from src.data.loader import (
    load_test,
    load_train,
    gap_mask,
)
from src.models.climatology import Climatology
from src.models.interpolation import (
    MLStacker,
    fill_series,
)


def _get_climatology() -> Climatology:
    try:
        return Climatology.load()
    except Exception:
        try:
            train_df = load_train()
            clim = Climatology().fit(train_df)
            clim.save()
            return clim
        except Exception as exc:
            raise RuntimeError(
                "Нет климатологии и не удалось обучить её по train_dataset.csv"
            ) from exc


def main(test_path=None, out_path=None) -> str:
    """
    Формирует submission.csv для строк:
        is_synthetic_gap = True
    """
    test_df = load_test(test_path)

    climatology = _get_climatology()

    try:
        model = MLStacker.load()
    except Exception:
        model = None

    filled_df = fill_series(test_df, climatology, model)

    mask = gap_mask(filled_df)
    gaps = filled_df[mask].copy()

    if gaps.empty:
        print(
            "WARNING: в тестовом датасете нет строк "
            "с is_synthetic_gap=True"
        )

        submission = pd.DataFrame(
            columns=[
                ID_COL,
                DATE_COL,
                "primary_ndvi_pred",
            ]
        )
    else:
        submission = pd.DataFrame(
            {
                ID_COL: gaps[ID_COL],
                DATE_COL: gaps[DATE_COL].dt.strftime("%Y-%m-%d"),
                "primary_ndvi_pred": gaps["primary_ndvi_filled"].round(6),
            }
        )

    submission["primary_ndvi_pred"] = (
        submission["primary_ndvi_pred"]
        .fillna(0.5)
        .clip(0, 1)
    )

    submission = submission.dropna()

    if submission.duplicated(subset=[ID_COL, DATE_COL]).any():
        submission = submission.drop_duplicates(
            subset=[ID_COL, DATE_COL],
            keep="last",
        )

    out = Path(out_path or SUBMISSION_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(
        out,
        index=False,
        encoding="utf-8",
    )

    print(f"submission сохранён: {out}, строк: {len(submission)}")

    return str(out)


if __name__ == "__main__":
    main()