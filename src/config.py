from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_dataset.csv"
TEST_CSV = DATA_DIR / "private_features.csv"
SUBMISSION_PATH = ROOT / "submission.csv"

ID_COL = "anon_polygon_id"
DATE_COL = "date"
TARGET = "primary_ndvi"
GAP_FLAG = "is_synthetic_gap"

RANDOM_SEED = 42

# --- GapScore ---
RMSE_THRESHOLD = 0.10
MAX_GAP_SCORE = 30

# --- Climatology ---
CLIM_WINDOW = 15

# --- Z-score anomalies ---
Z_NORMAL = -1.0
Z_STRESS = -2.0

MIN_ANOMALY_LEN = 2
MAX_PERIOD_GAP_DAYS = 20