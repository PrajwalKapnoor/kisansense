"""Central paths and constants so every module agrees on where things live."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"

# Expected raw dataset locations
CROP_CSV = RAW_DIR / "crop" / "Crop_recommendation.csv"
PRICE_CSV = RAW_DIR / "price" / "mandi_prices.csv"
DISEASE_DIR = RAW_DIR / "disease"

RANDOM_SEED = 42

# Crop recommendation dataset: expected schema
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_TARGET = "label"

# Plausible input ranges, used for API/UI validation
CROP_FEATURE_RANGES = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),
    "temperature": (0, 50),
    "humidity": (10, 100),
    "ph": (3, 10),
    "rainfall": (20, 300),
}

IMAGE_SIZE = (224, 224)
