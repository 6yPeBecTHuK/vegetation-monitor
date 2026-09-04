import numpy as np

from src.config import RMSE_THRESHOLD, MAX_GAP_SCORE


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return 0.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gap_score(rmse_value: float) -> float:
    """
    GapScore = round(30 * max(0, 1 - RMSE / 0.10), 2)
    """
    score = MAX_GAP_SCORE * max(0.0, 1.0 - rmse_value / RMSE_THRESHOLD)
    return round(float(score), 2)