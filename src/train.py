import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET,
    ID_COL,
    RANDOM_SEED,
)
from src.data.loader import load_train, gap_mask
from src.models.climatology import Climatology
from src.models.interpolation import (
    build_features,
    MLStacker,
)
from src.evaluation import rmse, gap_score


def make_synthetic_gaps(
    df: pd.DataFrame,
    frac: float = 0.15,
    exclude_mask: np.ndarray = None,
) -> pd.Series:
    """
    Создаёт дополнительные скрытые точки для обучения.
    """
    hide = pd.Series(False, index=df.index)

    if df.empty:
        return hide

    if exclude_mask is None:
        exclude_mask = np.zeros(len(df), dtype=bool)

    rng = np.random.default_rng(RANDOM_SEED)

    for _, g in df.groupby(ID_COL, sort=False):
        idx = g.index.to_numpy()

        candidates = idx[~exclude_mask[idx]]

        if len(candidates) == 0:
            continue

        n_hide = max(1, int(len(candidates) * frac))
        n_hide = min(n_hide, len(candidates))

        picked = rng.choice(
            candidates,
            size=n_hide,
            replace=False,
        )

        hide.loc[picked] = True

    return hide


def main():
    df = load_train()

    if df.empty:
        raise SystemExit("train_dataset.csv пуст или не загружен")

    print(
        f"train: {len(df)} строк, "
        f"{df[ID_COL].nunique()} полигонов"
    )

    # 1. Климатология
    clim = Climatology().fit(df)
    clim.save()
    print("Климатология сохранена")

    # 2. Скрытые точки: контрольные пропуски + аугментация
    original_hidden = gap_mask(df)

    hide = pd.Series(False, index=df.index)
    hide.iloc[np.where(original_hidden)[0]] = True

    extra_hidden = make_synthetic_gaps(
        df=df,
        frac=0.15,
        exclude_mask=original_hidden,
    )

    hide = hide | extra_hidden

    y_observed = df[TARGET].where(~hide).to_numpy(dtype=float)

    X, indices = build_features(df, clim, y_observed)

    if len(X) == 0:
        raise SystemExit("Не удалось построить обучающие скрытые точки")

    y_true = df.loc[indices, TARGET].to_numpy(dtype=float)

    print(f"Обучающих скрытых точек: {len(X)}")

    # 3. Валидация и сравнение с бейзлайном
    if len(X) >= 20:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_true,
            test_size=0.2,
            random_state=RANDOM_SEED,
        )

        model = MLStacker().fit(X_train, y_train, X_val, y_val)

        pred_val = model.predict(X_val)

        rmse_ml = rmse(y_val, pred_val)
        score_ml = gap_score(rmse_ml)

        baseline = (
            X_val["idw"]
            .fillna(X_val["clim_mean"])
            .to_numpy(dtype=float)
        )

        rmse_baseline = rmse(y_val, baseline)
        score_baseline = gap_score(rmse_baseline)

        print(f"Baseline IDW: RMSE={rmse_baseline:.4f}, GapScore={score_baseline}")
        print(f"ML model    : RMSE={rmse_ml:.4f}, GapScore={score_ml}")
    else:
        print("Слишком мало данных для отдельной валидации")

    # 4. Финальное обучение на всех скрытых точках
    final_model = MLStacker().fit(X, y_true)
    final_model.save()

    print("Модель сохранена в artifacts/stacker.pkl")


if __name__ == "__main__":
    main()