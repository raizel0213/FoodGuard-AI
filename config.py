"""FoodGuard AI - central configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "IndianFoodDelivery.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "foodguard_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_URL = (
    "https://raw.githubusercontent.com/YBIFoundation/MachineLearning/"
    "refs/heads/main/Dataset/IndianFoodDelivery.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_SPLITS = 5

TARGET = "delivery_issue_reported"
ID_COLS = ["order_id"]                      # identifier only - excluded
POST_EVENT_COLS = ["delivery_time_minutes"] # known only after delivery - excluded (leakage)

NUMERIC_FEATURES = [
    "distance_km",
    "order_value_inr",
    "traffic_index",
    "rainfall_mm",
    "restaurant_prep_est_min",
    "orders_ahead",
    "rider_experience_months",
    "is_peak_hour",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["city", "vehicle_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Risk bands: probability < upper_bound -> label
RISK_BANDS = [
    (0.10, "Low"),
    (0.20, "Medium"),
    (0.35, "High"),
    (float("inf"), "Critical"),
]

RISK_COLORS = {
    "Low": "#34d399",
    "Medium": "#fbbf24",
    "High": "#fb923c",
    "Critical": "#f87171",
}

THEME = {
    "text": "#e6f1ff",
    "muted": "#94a3b8",
    "accent": "#22d3ee",
    "accent2": "#0ea5e9",
    "grid": "rgba(148, 163, 184, 0.15)",
}
